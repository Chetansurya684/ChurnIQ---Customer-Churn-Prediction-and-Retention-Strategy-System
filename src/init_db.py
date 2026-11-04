from db import test,init
ok,msg=test(); print(msg)
if ok: init(); print('Tables initialized.')
else: print('Start MySQL and create churn_db first.')
