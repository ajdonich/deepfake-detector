import psycopg2
import config.config as dfc

class VideoTuple:
#{
    def __init__(self, video_id='DEFAULT', blk_id=None, split=None, vidname=None, 
                 part_id=None, label=None, origname=None, preprocflg=False):
    #{
        self.video_id = video_id
        self.blk_id = blk_id
        self.split = split

        self.vidname = vidname
        self.part_id = part_id
        self.label = label
        self.origname = origname
        self.preprocflg = preprocflg        
    #}

    def __repr__(self):
        return (f"VideoTuple: {self.video_id}, {self.blk_id}, {self.split}, {self.vidname}, "
                f"{self.part_id}, {self.label}, {self.origname}, {self.preprocflg}")

    def get_tuple(self):
        return (self.video_id, self.blk_id, self.split, self.vidname, 
                self.part_id, self.label, self.origname, self.preprocflg)
#}

class EpochTuple:
#{
    def __init__(self, epoch_id, blk_id, status):
        self.epoch_id = epoch_id
        self.blk_id = blk_id
        self.status = status

    def __repr__(self):
        return (f"EpochTuple: {self.epoch_id}, {self.blk_id}, {self.status}")

    def get_tuple(self):
        return (self.epoch_id, self.blk_id, self.status)
#}

class PostgreSqlHandle: 
#{
    _epoch_schema = (
        """ CREATE TABLE epoch_queue (
                epoch_id SERIAL PRIMARY KEY
                blk_id INTEGER NOT NULL,
                status VARCHAR(16) NOT NULL
        """)
    
    _videos_schema = (
        """ CREATE TABLE videos (
                video_id SERIAL PRIMARY KEY,
                blk_id INTEGER NOT NULL,
                split VARCHAR(8) NOT NULL,
                vidname VARCHAR(32) NOT NULL,
                part_id INTEGER NOT NULL,
                label VARCHAR(8) NOT NULL,
                origname VARCHAR(32),
                proc_flg BOOLEAN NOT NULL)
        """)

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

    def sqlquery(self, sql, fetch='all'):
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
            
            print("Commencing database initialization...")
            if vexists: self.cursor.execute("DROP TABLE videos")
            if eexists: self.cursor.execute("DROP TABLE epoch_queue")
            self.cursor.execute(PostgreSqlHandle._epoch_schema)
            self.cursor.execute(PostgreSqlHandle._videos_schema)
            self.dbconnection.commit()
            print("Database initialization complete.")

        #}
        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error); return False
        
        return True
    #}

    def populate_database(self, vtrains, vvalids):
    #{
        #ebsql = "INSERT INTO epoch_queue VALUES(%s, %s, %s, %s, %s)"
        vsql = "INSERT INTO videos VALUES(%s, %s, %s, %s, %s, %s, %s, %s);"

        try:
            print("Commencing database population...")
            #for eb in eblocks: self.cursor.execute(ebsql, eb.get_tuple())
            for vt in vtrains: self.cursor.execute(vsql, vt.get_tuple())
            for vv in vvalids: self.cursor.execute(vsql, vv.get_tuple())
            self.dbconnection.commit()
            print("Database population complete.\n")

        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR:", error)
    #}
#}
