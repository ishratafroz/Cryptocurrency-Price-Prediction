import os
from deta import Deta
from dotenv import load_dotenv
load_dotenv(".env")
DETA_KEY=os.getenv("DETA_KEY")
deta=Deta(DETA_KEY)
db=deta.Base("users_db")
def insert_user(username,name,password):
    """Returns the user on a successful user creation,otherwse raises and error"""
    return db.put({"key":username,"name":name,"password":password})
#insert_user("pparker","Peter Parker","abc123")
def fetch_all_users():
    """Returns a dict of all users """   
    res=db.fetch()
    return res.items
print(fetch_all_users())
def get_user(username):
    """if not found,the fucntion wll return none"""
    return db.get(username)
def update_user(username,updates):
    """if the item is updated, returns none.otherwise an excepton is raised"""
    return db.update(updates,username)
def delete_user(username):
    return db.delete(username)
