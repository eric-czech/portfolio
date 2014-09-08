import boto

def describe_instances():
    con = boto.connect_ec2()
    instances = con.get_all_instances()
    for instance in instances:
	    print instance
    con.close()

if __name__ == "__main__":
    describe_instances()
