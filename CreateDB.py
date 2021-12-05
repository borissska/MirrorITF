import os
import psycopg2
import psycopg2 as db

DATABASE_URL = os.environ["postgres://xhcapgvloicexy:276dab641de7c77214bf42fd0afc1f13ce8f4ee025c06b871325767572ecbc9b@ec2-34-242-89-204.eu-west-1.compute.amazonaws.com:5432/d7ed08ab2v640"]
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()

cursor.execute("CREATE TABLE instructor (ID CHAR(5),name VARCHAR(20) NOT NULL,dept_name VARCHAR(20),salary NUMERIC(8,2),PRIMARY KEY (ID))")

#cursor.execute("create table (personID VARCHAR(20) NOT NULL, generalPose , <col_name3><col_type3> PRIMARY KEY(<col_name1>), FOREIGN KEY(<col_name2>) REFERENCES <table_name2>(<col_name2>))")

cursor.close()
conn.close()