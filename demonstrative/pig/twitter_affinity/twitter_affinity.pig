-- Twitter Affinity Script
--
-- This purpose of this script is to determine the "affinity" between any two 
-- twitter users.  Defined as the sum of the number of followers of unique users 
-- mentioning BOTH users in any user pair at least once, this metric roughly 
-- estimates the size of the audience witness to discussions about two entities
-- that are often discussed by the same people.
--
-- Presumably, the higher this score is for any user pair, the "stronger" the relationship
-- is between the two.  Or at the very least, a higher score indicates that the same
-- people like to talk about both, regardless of what the relationship between
-- them actually is.
--
-- Example invocations:
-- pig -f twitter_affinity.pig -param_file twitter_affinity.params -param INPUT_DIR=tweet_data OUTPUT_DIR=affinity_data


-- Set some primary job configuration options
SET job.name 'Twitter Affinity'
SET mapred.job.queue.name 'analysis'

-- Define the input/output paths and remove any old output, if it exists
%DEFAULT OUTPUT_DIR 'affinity_data'
%DEFAULT INPUT_DIR 'tweet_data'
fs -rm -r -f $OUTPUT_DIR

-- Determine the sets of user pairs to focus on (params are comma-separated handle names)
-- * Result produced will contain cross product of these sets where affinity > 0
%DEFAULT GROUP1 '$DEFAULT_GROUP1'
%DEFAULT GROUP2 '$DEFAULT_GROUP2'

-- UDFs here will split out input csv string elements and act as filter for raw data
DEFINE IsInGroup1 com.nextbigsound.hadoop.common.pig.udf.InUDF('$GROUP1');
DEFINE IsInGroup2 com.nextbigsound.hadoop.common.pig.udf.InUDF('$GROUP2');

raw = LOAD '$INPUT_DIR' AS ($DATA_SCHEMA);
-- raw: {
--   actor_handle: charrary,   # username of tweeter
--   target_handle: chararray, # username of tweetee
--   actor_followers: long     # number of followers the actor has
-- }

-- Split the tweet data into separate groups based on which input 
-- set the handle being mentioned falls in
SPLIT raw INTO group1 IF IsGroup1(target_handle), group2 IF IsGroup2(target_handle);

-- Now group by the actor to give two sets per user, one with who that
-- person mentioned in group1 and one with who that person mentioned in group2
byUser = COGROUP group1 BY actor_handle, group2 BY actor_handle;

-- Determine the cross product of target users for each actor
crossProduct = FOREACH byUser GENERATE FLATTEN(group1), FLATTEN(group2);
-- crossProduct: {
--   # Note that both "actor_handle" values will be equal for each tuple
--   group1::actor_handle: chararray,
--   group1::target_handle: chararray,
--   group1::actor_followers: long,
--   group2::actor_handle: chararray,
--   group2::target_handle: chararray,
--   group2::actor_followers: long
-- }
	
-- Group the cross product by each occurrence of target handles in the input sets
pairs = GROUP crossProduct BY (group1::target_handle, group2::target_handle);

-- Finally, compute the some of the follower counts for each user that 
-- mentioned both target handles in each group
affinities = FOREACH pairs GENERATE FLATTEN(group), SUM(crossProduct.group1::actor_followers) AS affinity;
-- affinities: { 
--   group1::target_handle: chararray, 
--   group2::target_handle: chararray,
--   affinity: long
-- }

result = ORDER affinities BY affinity DESC;
STORE result INTO '$OUTPUT_DIR';

