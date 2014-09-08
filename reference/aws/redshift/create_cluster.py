
import boto

def create_cluster():
	con = boto.redshift.layer1.RedshiftConnection()
	return con.create_cluster(
            cluster_identifier='ext-account-sony',
            node_type='dw2.large',
            master_username='sony',
            master_user_password='dBDSqEV75m5J3YFsCQnRsYyg',
            db_name='snapshot20140818',
            cluster_type='single-node',
            cluster_security_groups='sony',
            vpc_security_group_ids='None',
            cluster_subnet_group_name='None',
            availability_zone='us-east-1',
            preferred_maintenance_window='None',
            cluster_parameter_group_name='None',
            automated_snapshot_retention_period='None',
            port=5439,
            allow_version_upgrade=True,
            number_of_nodes=1,
            publicly_accessible=False,
            encrypted=True)

if __name__ == "__main__":
	create_cluster()
