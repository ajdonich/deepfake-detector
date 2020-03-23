import psycopg2
import config.config as dfc

class VideoTuple:
#{
    insertproto = ("INSERT INTO videos VALUES(DEFAULT, "
                   "{}, '{}', '{}', {}, '{}', '{}', {})")
    createsql = ("""CREATE TABLE videos (
                        video_id SERIAL PRIMARY KEY,
                        blk_id INTEGER NOT NULL,
                        split VARCHAR(8) NOT NULL,
                        vidname VARCHAR(32) NOT NULL,
                        partition INTEGER NOT NULL,
                        label VARCHAR(8) NOT NULL,
                        origname VARCHAR(32),
                        proc_flg BOOLEAN NOT NULL);
                    CREATE INDEX ON videos (blk_id, split)
                 """)

    def __init__(self, video_id=None, blk_id=None, split=None, vidname=None, 
                 partition=None, label=None, origname=None, preprocflg=False):
    #{
        self.video_id = video_id
        self.blk_id = blk_id
        self.split = split

        self.vidname = vidname
        self.partition = partition
        self.label = label
        self.origname = origname
        self.preprocflg = preprocflg

        self._insertsql = None
    #}

    def __repr__(self):
        return (f"VideoTuple: {self.video_id}, {self.blk_id}, {self.split}, {self.vidname}, "
                f"{self.partition}, {self.label}, {self.origname}, {self.preprocflg}")

    @property
    def insertsql(self): 
        return VideoTuple.insertproto.format(
            self.blk_id, self.split, self.vidname, self.partition, 
            self.label, self.origname, self.preprocflg)
#}

class EpochTuple:
#{
    insertproto = "INSERT INTO epoch_queue VALUES(DEFAULT, {}, '{}')"
    createsql = (""" CREATE TABLE epoch_queue (
                         epoch_id SERIAL PRIMARY KEY,
                         blk_id INTEGER NOT NULL,
                         status VARCHAR(16) NOT NULL)
                 """)

    def __init__(self, epoch_id=None, blk_id=None, status=None):
        self.epoch_id = epoch_id
        self.blk_id = blk_id
        self.status = status
        self._insertsql = None

    def __repr__(self): 
        return (f"EpochTuple: {self.epoch_id}, {self.blk_id}, {self.status}")

    @property
    def insertsql(self): 
        return EpochTuple.insertproto.format(self.blk_id, self.status)
#}

class PostgreSqlHandle: 
#{
    def __init__(self, verbose=False): 
        self._cursor = None
        self.dbconnection = None
        self.verbose = verbose
    
    def __enter__(self):
    #{
        self.dbconnection = None
        try:
        #{
            if self.verbose: print('Connecting to the PostgreSQL database...')
            self.dbconnection = psycopg2.connect(host=dfc.HOST, database=dfc.DATABASE,
                user=dfc.DBUSER, password=dfc.DBPASSWORD, port=dfc.DBPORT)
            
            # Validate connection
            self.cursor = self.dbconnection.cursor()
            self.cursor.execute('SELECT version()')
            version = self.cursor.fetchone()
            if self.verbose: print(f"PostgreSQL version:\n  {version}")
        #}  
        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error)

        return self
    #}
    
    def __exit__(self, exception_type, exception_value, traceback):
    #{
        if self.cursor is not None:
            self.cursor.close(); self.cursor = None
        
        if self.dbconnection is not None:
            self.dbconnection.close(); self.dbconnection = None
    #}

    @property
    def cursor(self):
        if self._cursor is None and self.dbconnection is not None:
            self.cursor = self.dbconnection.cursor()
        return self._cursor

    @cursor.setter
    def cursor(self, value): 
        self._cursor = value 

    def sqlquery(self, sql, fetch=None):
    #{
        result = None
        try:
            self.cursor.execute(sql)
            if fetch == 'all': result = self.cursor.fetchall()
            elif fetch == 'one': result = self.cursor.fetchone()
            elif fetch is not None: result = self.cursor.fetchmany(fetch)
            self.dbconnection.commit()

        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error)

        return result
    #}

    def initialize_database(self):
    #{
        try:
        #{
            eexists = self.sqlquery("SELECT to_regclass('epoch_queue')", fetch='one')[0] is not None
            vexists = self.sqlquery("SELECT to_regclass('videos')", fetch='one')[0] is not None

            if eexists:
                count = self.sqlquery("SELECT COUNT(*) FROM epoch_queue", fetch='one')[0]
                print(f"WARNING: table 'epoch_queue' already exists with {count} rows.")
            
            if vexists:
                count = self.sqlquery("SELECT COUNT(*) FROM videos", fetch='one')[0]
                print(f"WARNING: table 'videos' already exists with {count} rows.")

            usrrsp = 'y'
            if eexists or vexists: usrrsp = input(f"Are you sure you want reinitialize this database?\n[N/y]")
            if usrrsp.lower() != 'y': print("Database initialization operation aborted."); return False
            
            print("\nCommencing database initialization...")
            if vexists: self.cursor.execute("DROP TABLE videos")
            if eexists: self.cursor.execute("DROP TABLE epoch_queue")
            self.cursor.execute(EpochTuple.createsql)
            self.cursor.execute(VideoTuple.createsql)
            self.dbconnection.commit()
            print("Database initialization complete.")
        #}
        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error); return False
        
        return True
    #}

    def populate_database(self, vtrains, vvalids):
    #{
        try:
            print("\nCommencing database population...")
            for vt in vtrains: self.cursor.execute(vt.insertsql)
            for vv in vvalids: self.cursor.execute(vv.insertsql)
            self.dbconnection.commit()
            print("Database population complete.\n")

        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error)
    #}
#}
