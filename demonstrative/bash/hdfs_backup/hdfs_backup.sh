#!/bin/bash

# HDFS file backup script used to download immutable files from Hadoop 
# into one of several local directories as unrepeated hard-links.
#
# Backups are created by downloading all files to a "master" directory and then 
# hard-linking those files to one of several "archive" directories.  These archive
# directories are set as the current day, week, and month, where each is never 
# overwritten after the initial linking (creating separate retention frequencies). 
#
# The number of "archive" directories then retained for each frequency is controlled using
# the "retention" parameter which specifies the maximum number of most recent directories
# to keep (older ones are deleted)
#
# Example invocations:
# hdfs_backup.sh -p /backups/hadoop/data -r 12 -s /hadoop/data >> ~/hdfs_backup.log/logs 2>&1
#
# Author: Eric Czech

usage () {
	[ -n "$1" ] && echo "ERROR: $1" >&2

	cat <<-EOF >&2

		Usage: $BASH_SOURCE -p <backup path> -s <hdfs path> -r <retention> [-d] [-f]
		       $BASH_SOURCE -h|--help

		  -p <backup path> # Path to backup directory on local file system
		  -r <retention>   # Number of folders to keep for each time period (eg 7 would mean keep 7 daily backups, 7 weekly, and 7 monthly)
		  -s <source path> # Path to HDFS directory to backup
		  -d               # dryrun - specifies that no files will be changed, all operations will be printed to stdout instead
		  -f		   # force - causes any existing backups to be overwritten for each retention frequency
		  -h|--help        # print this help and exit

	EOF

	exit 1
}

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
}

while getopts p:r:s:fdh OPT
do
	case $OPT in
		p) BACKUP_DIR="$OPTARG" ;;
		r) RETENTION="$OPTARG" ;;
		d) DRY_RUN="--dry-run" ;;
		f) FORCE_BACKUP="--force" ;;
		s) HDFS_DIR="$OPTARG" ;;
		h) usage ;;
		*) usage ;;
	esac
done

shift $(( $OPTIND -1 ))

# Validate input arguments
[ -z "$BACKUP_DIR" ] && usage "Missing required argument: backup path"
[ -z "$RETENTION" ] && usage "Missing required argument: retention"
[ -z "$HDFS_DIR" ] && usage "Missing required argument: hdfs path"

echo "Starting HDFS backup for path = $HDFS_DIR, target = $BACKUP_DIR, retention = $RETENTION (date = `date`)"

MASTER_DIR=$BACKUP_DIR/master
ARCHIVE_DIR=$BACKUP_DIR/archives

# Set archive directory names for the current day, week, and month
A_DIRS[0]="$ARCHIVE_DIR/daily/`date +'%Y-%m-%d'`"
A_DIRS[1]="$ARCHIVE_DIR/weekly/`date +'%Y-%W'`"
A_DIRS[2]="$ARCHIVE_DIR/monthly/`date +'%Y-%m'`"

# Create the master directory
if [ ! -d $MASTER_DIR ]; then
	mkdir -p $MASTER_DIR
	if [ $? -ne 0 ]; then
		err "Failed to create backup directory $MASTER_DIR"
		exit 1
	fi
fi

DATA_DIRS=()
# Create each archive directory if it does not exist
for dir in "${A_DIRS[@]}" 
do
	if [ ! -d $dir ]; then
		mkdir -p $dir
		if [ $? -ne 0 ]; then
			err "Failed to create backup directory $BACKUP_DIR"
			exit 1
		fi
	else
		# If force option is given and directory exists, remove contents 
		if [ -n "$FORCE_BACKUP" ]; then
			if [ `ls $dir | wc -l` -gt 0 ] && [ -z "$DRY_RUN" ]; then
				echo "Removing old links in backup dir $dir"
				rm $dir/*
				if [ $? -ne 0 ]; then
					err "Failed to remove old links in dir $dir" 
					exit 1
				fi
			fi
		else
			echo "Ignoring existing archive dir $dir"
			continue
		fi
	fi
	echo "Adding archive dir $dir as backup target"
	DATA_DIRS+=($dir)
done

# Clean up old directories as determined by $RETENTION
for dir in "${DATA_DIRS[@]}" 
do
	ARCHIVE=`dirname $dir`
	DIRS=(`ls $ARCHIVE | sort -r`)
	for OLD_DIR in "${DIRS[@]:$RETENTION}"
	do
		echo "Removing old archive directory $ARCHIVE/$OLD_DIR"
		[ -z "$DRY_RUN" ] && rm $ARCHIVE/$OLD_DIR/*
		[ -z "$DRY_RUN" ] && rmdir $ARCHIVE/$OLD_DIR
	done
done

# Log input options
echo "Starting HDFS backup to dir '$BACKUP_DIR' from HDFS dir '$HDFS_DIR' with retention $RETENTION (`date`)"
if [ -n "$FORCE_BACKUP" ]; then
	echo "* Forcing overwrite of old archive directories (-f passed)"
fi
if [ -n "$DRY_RUN" ]; then
	echo "* Dry run (-d passed)"
fi

# For each file in the Hadoop directory, download to the master directory
# and then link to the appropriate archive folders
for FILE_PATH in `hadoop fs -ls $HDFS_DIR | awk '{print $8}'`
do
	file=`basename $FILE_PATH`
	if [ ! -f $MASTER_DIR/$file ]; then
		echo "Downloading: $file to $MASTER_DIR/"
		[ -z "$DRY_RUN" ] && hadoop fs -get $HDFS_DIR/$file $MASTER_DIR/
	fi
	for dir in "${DATA_DIRS[@]}"
	do
		echo "Linking $MASTER_DIR/$file to $dir/$file"
		if [ -z "$DRY_RUN" ]; then
			ln $MASTER_DIR/$file $dir/$file
			if [ $? -ne 0 ]; then
				err "Failed to link $MASTER_DIR/$file to $dir/$file (moving on anyways)" 
			fi
		fi
	done
done

echo "Backup complete (date = `date`)"
exit 0
