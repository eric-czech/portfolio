-- NBS 15 chart calculation script
--
-- Execution outline:
-- 1. Load Billboard 200 score data by entity
-- 2. Load previous Billboard chart appearances from Mongo
-- 3. Load entities to remove based on manual exceptions
--      indicated in the past or those that are not musicians
-- 4. Join score data to entities that have charted before or 
--      that should be ignored for some other reason
-- 5. Filter score data to only desirable entities
-- 6. Order the score data by BB 200 likelihood value and limit
--      to only the top N entities
-- 7. Store the resulting "chart" in MySQL with one final filter
--    applied to remove artists that have chart appearances per
--    Billboard.com (search there is an HTML scrape by artist name)
--
-- Note that the final filter step (the Billboard.com check) is combined with the
-- storage step (using NBS15StoreFunction) so that the concurrency of calls to the 
-- Billboard.com site can be controlled (via STORAGE_PARALLEL).  Every call to their site 
-- is slow so the concurrency here should always be relatively high (~10).

@PIG_DEFAULTS
@ANALYTICS_DEFAULTS
@DATE_CONSTANTS
@LIPSTICK
@MONGO_DEFAULTS

SET mapred.job.queue.name 'critical'
SET job.name 'NBS 15 Calculator'

----------------------------
-- Data filter parameters --
----------------------------

-- Index containing BB 200 scores
%DECLARE INDEX_NAME 'idx_entity'

-- This command selects the most recent HBase extract for the index by sorting by the timestamp in the directory names 
%DECLARE INPUT_DIR `hadoop fs -ls $EXTRACT_DATA_DIR/$INDEX_NAME | grep '$INDEX_NAME' | awk '{ print $8 }' | sort -t'/' -k 6 -r | head -n 1`

-- Unix week timestamp associated with this chart (weeks are Monday to Sunday)
%DEFAULT WEEK_TIMESTAMP `echo $(($(date +%s)/604800))`

-- Unix seconds timestamp for start of week (a Monday at 00:00)
%DECLARE START_SECONDS '$WEEK_TIMESTAMP * $SECONDS_IN_A_WEEK - 3 * $SECONDS_IN_A_DAY'

-- Unix seconds timestamp for end of week (a Sunday at 23:59)
%DECLARE STOP_SECONDS  '$WEEK_TIMESTAMP * $SECONDS_IN_A_WEEK + 4 * $SECONDS_IN_A_DAY - 1'

-----------------------------
-- Chart filter parameters --
-----------------------------

-- "Context" for loading of entity names and previous NBS 15 charts
%DEFAULT SQOOP_CONTEXT '$CURRENT_DATE'

-- Mongo query document determining what previous chart appearances 
-- should be used to exclude entities from the NBS chart
%DECLARE CHART_QUERY '{ "chart_crawler_name_id" : 6 }'
--%DECLARE CHART_QUERY '{ "chart_name_id" : { \$in: [ 125 ] } }'


-----------------------------
-- Chart result parameters --
-----------------------------

-- Determines the "initial" limit on chart entries to be later filtered 
-- further using checks for chart appearances via Billboard.com;
-- Note that a limit here of around 1500 will reduce to roughly 240 final
-- entries after the Billboard check (~16% acceptance rate)
%DEFAULT INIT_CHART_LIMIT 1500

-- Determines how many simultaneous jobs will run to store chart entries
-- only after checking them against Billboard.com (which is slow and
-- requires this concurrency)
%DEFAULT STORAGE_PARALLEL 10

---------------------
-- Load Score Data --
---------------------
-- Loads HBase snapshot data containing Billboard 200 likelihood scores

scores_raw = LOAD '$INPUT_DIR' USING parquet.pig.ParquetLoader;
DESCRIBE scores_raw;
-- scores_raw: {
--   account_id: int, entity_id: int, endpoint_id: int, metric_id: int,
--   count_type: chararray, unix_seconds: long, value: double
-- }

scores_filtered = FILTER scores_raw BY 
	metric_id == 310 -- BB 200 Score metric
	AND unix_seconds >= $START_SECONDS 
	AND unix_seconds <= $STOP_SECONDS
	AND count_type == 'd'
	AND value < 100;
	
--popular_entities_raw = FILTER scores_raw BY 
--	metric_id == 11 
--	AND count_type == 't'
--	AND unix_seconds >= $START_SECONDS 
--	AND unix_seconds <= $STOP_SECONDS
--	AND value > 500000;
--popular_entities = DISTINCT (FOREACH popular_entities_raw GENERATE entity);
--
--applicable_entity_val = FOREACH (GROUP applicable_entity_raw BY entity) GENERATE 
--	group AS entity, MAX(applicable_entity_raw.value) AS value;
--applicable_entity_filtered = FILTER applicable_entity_val BY value < 500000;
--applicable_entity_filtered


scores_by_entity = GROUP scores_filtered BY (entity_id, metric_id);

scores = FOREACH scores_by_entity {
  i1 = ORDER scores_filtered BY unix_seconds DESC;
  i2 = LIMIT i1 1;
  GENERATE FLATTEN(group), FLATTEN(i2.(value));
}
DESCRIBE scores;


---------------------------------
-- Load Known Charted Entities --
---------------------------------
-- Loads any known Billboard chart appearances using event data in MongoDB
-- where the relevant documents are selected using CHART_QUERY


DEFINE ExternalChartLoader com.nextbigsound.hadoop.common.mongo.MongoLoader(
	'$MONGO_PRIMARY', '$CHART_QUERY', '{ "artist_id":1, "day":1 }'
);

-- Load existing chart appearances as defined by CHART_QUERY
ext_charts_raw = LOAD 'charts.crawled_charts' Using ExternalChartLoader AS (record:map[chararray]);

-- Compute week timestamp for each appearance
ext_chart_dates = FOREACH ext_charts_raw GENERATE 
	(int)record#'artist_id' AS entity_id:int,
	((int)record#'day') / 7 AS week:int;

-- Get the first week timestamp of appearance for each entity
ext_chart_by_entity = GROUP ext_chart_dates BY (entity_id);

ext_charts = FOREACH ext_chart_by_entity GENERATE group AS entity_id, MIN(ext_chart_dates.week) AS week;
DESCRIBE ext_charts;

----------------------------------
-- Load Known Entity Exceptions --
----------------------------------
-- Pulls in entity ids to ignore for those that are not musicians
-- or that have been explicitly marked as exceptions in the past

DEFINE InternalChartLoader com.nextbigsound.hadoop.common.pig.udf.SqoopLoader(
	'-context', '$SQOOP_CONTEXT'
);

-- Load artists explicitly marked as 'Not Applicable' for the NBS 15 chart
int_charts_raw = LOAD 'NBS15_CHART' USING InternalChartLoader() AS (
	entity_id: int, position: int, week_timestamp: int, active:chararray
);

-- Create distinct list of excluded entity ids 
int_charts_filtered = FILTER int_charts_raw BY LOWER(active) == 'no';
int_charts_proj = FOREACH int_charts_filtered GENERATE entity_id;
int_charts_dist = DISTINCT int_charts_proj;
int_charts = FOREACH int_charts_dist GENERATE entity_id, (int)null AS week:int;
DESCRIBE int_charts;


-- Load x_artists_categories to create a list of non-music entities for removal
entity_cat = LOAD '$META_WAREHOUSE/meta_x_artists_categories' AS (
	entity_id:int, category_id:int, rank_value:long, created_at:chararray, updated_at:chararray, deleted_at:chararray
);

entity_cat_filtered = FILTER entity_cat BY deleted_at == '\\N';
entity_cat_group = GROUP entity_cat_filtered BY (entity_id);

-- Select the "primary" category for the entity using the "rank" associated with it
entity_cat_top = FOREACH entity_cat_group {
	i1 = ORDER entity_cat_filtered BY rank_value DESC;
	i2 = LIMIT i1 1;
	GENERATE group AS entity_id, FLATTEN(i2.category_id) AS category_id;
}

-- Finally, create the list of non-music entities
entity_cat_non_music = FILTER entity_cat_top BY category_id != 1; -- category id 1 is 'Music'
non_music_entities = FOREACH entity_cat_non_music GENERATE entity_id, (int)null AS week:int;
DESCRIBE non_music_entities;

---------------------------------------------
-- Exclude Undesirable Entities from Chart --
---------------------------------------------

-- Create unified list of entries to (potentially) exclude
entity_filter = UNION ext_charts, int_charts, non_music_entities;

-- Outer join score data to excluded entity list 
joined = JOIN scores BY entity_id LEFT OUTER, entity_filter BY entity_id;
DESCRIBE joined;


-- Remove score data for artists that have already charted or have been marked as chart exceptions to ignore
joined_filtered = FILTER joined BY
	-- Select records with no prior chart appearances
	entity_filter::entity_id IS NULL
	 
	-- Or records where the date of appearance is after the week being calculated (BB chart dates 
	-- are stored 1 week ahead of what they really represent so subtract 1 before checking) 
	OR (entity_filter::week IS NOT NULL AND entity_filter::week - 1 > $WEEK_TIMESTAMP);
	
result_proj = FOREACH joined_filtered GENERATE 
	scores::group::entity_id AS entity_id:int,
	value AS value:double,
	$WEEK_TIMESTAMP AS week_timestamp:int;
	
-- Order the result by score and take only the top INIT_CHART_LIMIT entries
result_ordered = RANK result_proj BY value DESC, entity_id DESC;
result_init = LIMIT ( FOREACH result_ordered GENERATE $0 AS position:long, entity_id, value, week_timestamp ) $INIT_CHART_LIMIT;
DESCRIBE result_init;

--------------------------------------------------
-- Store Results (after further filtering them) --
--------------------------------------------------

-- Load artist/entity names and join them to the current chart list
DEFINE EntityLoader com.nextbigsound.hadoop.common.pig.udf.SqoopLoader('-context', '$SQOOP_CONTEXT');
entity_names = LOAD 'META_ARTISTS' USING EntityLoader() AS (
	entity_id:int, entity_name:chararray, mbz_id:chararray, rank:long
);
result_joined = JOIN entity_names BY entity_id, result_init BY entity_id;

-- Create the final relation representing the chart
result = FOREACH result_joined GENERATE position, entity_names::entity_id, entity_name, week_timestamp;
DESCRIBE result;

-- Disable speculative execution for MySQL storage
SET mapreduce.map.speculative 'false'
SET mapreduce.reduce.speculative 'false'

-- Partition the chart entries in STORAGE_PARALLEL groups, processing them in that many separate reduce tasks
result_partition = GROUP result BY (position % $STORAGE_PARALLEL) PARALLEL $STORAGE_PARALLEL;

-- Store the resulting chart entries in MySQL; ChartStorage will first check that each artist has never
-- appeared on a Billboard chart via an HTML scrape before accepting it
DEFINE ChartStorage com.nextbigsound.hadoop.analytics.chart.NBS15StoreFunction();
STORE ( FOREACH result_partition GENERATE FLATTEN(result) ) INTO 'stats.chart' USING ChartStorage;

