#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random 
lower="abcdefghijklmnopqrstuvwxyz" 
upper="ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
number="0123456789" 
symbols="!@#$%^&*(),./'\][~`-:<>_+}{=|?" 
string=lower+upper+number+symbols

print ("Welcome!") #greet to the user 

counter=0

while counter==0:
    length= eval(input("Please enter your password length: ")) 
    if length>=8 and length<=16:
        password="".join(random.sample(string,length))
        print("Your password is:"+password)
        counter=1
    else: 
        print("The length must be between 8 to 16 digits.") 
        counter=0


# In[ ]:




