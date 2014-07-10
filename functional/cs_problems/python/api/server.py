from flask import Flask
from flask import request

app = Flask(__name__)

from most_consecutive_item import lib as mci
from most_frequent_item import lib as mfi

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def to_int_array(arr):
    arr_ints = [ int(x) if is_int(x) else None for x in arr.split(',') ]
    return [ x for x in arr_ints if x is not None ]

@app.route("/most_consecutive/<arr>")
def most_consecutive_item(arr):
    arr_ints = to_int_array(arr)
    res = mci.findMostConsecutivelyRepeatingValue_ideal(arr_ints)
    return 'Input Array = {}<br>Value with the most consecutive appearances = {}<br>Number of consecutive appearances = {}'\
        .format(arr_ints, res[0] if res else 'None', res[1] if res else 'None')

@app.route("/most_frequent/<arr>")
def most_frequent_item(arr):
    arr_ints = to_int_array(arr)
    res = mfi.findMostFrequentValue_ideal(arr_ints)
    return 'Input Array = {}<br>Most frequent value = {}<br>Number of appearances = {}'\
        .format(arr_ints, res[0] if res else 'None', res[1] if res else 'None')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_debugger=True, use_reloader=True, port = 8000 )
