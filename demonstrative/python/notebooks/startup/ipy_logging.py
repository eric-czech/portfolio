# In-notebook logging configuration
import sys
import logging
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console)

# If the first argument passed is the string "true", then remove the first logging handler.  
# This was necessary to make logging working in versions of jupyter earlier than 1.0.0
if len(sys.argv) > 1 and sys.argv[1] == 'true':
    logger.removeHandler(logger.handlers[0])
