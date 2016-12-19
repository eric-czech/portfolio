# In-notebook logging configuration
import sys
import logging
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console)

# If no arguments were passed or if the first argument does not equal "false"
# then remove the first logging handler.  This was necessary to make logging
# working in versions of jupyter earlier than 1.0.0
if len(sys.argv) == 1 or sys.argv[1] != 'false':
    logger.removeHandler(logger.handlers[0])
