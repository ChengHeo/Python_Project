#!/usr/bin/env python
# coding: utf-8

# In[1]:


print ("Welcome!") #greet to the user

import random
lower="abcdefghijklmnopqrstuvwxyz"
upper="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
number="0123456789"
symbols="!@#$%^&*(),./'\][~`-:<>_+}{=|?"
string=lower+upper+number+symbols

length= eval(input("Please enter your password length: "))
if length>=8 and length<=16:
    l=int(length)
else:
    print("The length must be between 8 to 16 digits.")
    length=eval(input("Please enter again: "))
    l=int(length)

password="".join(random.sample(string,l))
print("Your password is:"+password)


# In[3]:


print ("Welcome!") #greet to the user

import random
lower="abcdefghijklmnopqrstuvwxyz"
upper="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
number="0123456789"
symbols="!@#$%^&*(),./'\][~`-:<>_+}{=|?"
string=lower+upper+number+symbols

length= eval(input("Please enter your password length: "))
if length>=8 and length<=16:
    l=int(length)
else:
    print("The length must be between 8 to 16 digits.")
    length=eval(input("Please enter again: "))
    l=int(length)

password="".join(random.sample(string,l))
print("Your password is:"+password)


# In[ ]:




