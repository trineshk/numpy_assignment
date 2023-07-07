#!/usr/bin/env python
# coding: utf-8

# > ## **Assignment - Numpy Array Operations** 
# >
# > The objective of this assignment is to develop a solid understanding of Numpy array operations. In this assignment you will:
# > 
# > 1. Pick 5 interesting Numpy array functions by going through the documentation: https://numpy.org/doc/stable/reference/routines.html 
# > 2. Run and modify this Jupyter notebook to illustrate their usage (some explanation and 3 examples for each function). Use your imagination to come up with interesting and unique examples.
# > 3. Upload this notebook to your Jovian profile using `jovian.commit` and make a submission here: https://jovian.ai/learn/zero-to-data-analyst-bootcamp/assignment/assignment-4-exploring-numpy-functions
# > 4. (Optional) Share your notebook online (on Twitter, LinkedIn, Facebook) and with the course community.
# > 5. (Optional) Check out the notebooks shared by other participants and give feedback & appreciation.
# >
# > Try to give pick a theme for your assignment and give your notebook an interesting title e.g. "All about Numpy array operations", "5 Numpy functions you didn't know you needed", "A beginner's guide to broadcasting in Numpy", "Interesting ways to create Numpy arrays", "Trigonometic functions in Numpy", "How to use Python for Linear Algebra" etc.
# >
# > **NOTE**: Remove this cell containing explanations before submitting or sharing your notebook online - to make it more presentable.
# 
# 
# 

# # Frequently Used NumPy Functions in Machine Learning & Data Science
# 
# 
# **What is NumPy?**
# * NumPy stands for Numerical Python, a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
# 
# **Why NumPy?**
# * In Python we have lists that serve the purpose of arrays,but the are slow to process. NumpPy provide an array object that is up to 50X faster than traditional Python lists.
# 
# **How NumPy is faster than Lists?**
# * NumPy arrays are stored at one continuous place in memory unlike in lists, so processes can access and manipulate them very efficiently, This behavior is called Locality of Reference in computer science.

# In[3]:


get_ipython().system('pip install jovian --upgrade -q')


# In[4]:


import jovian


# In[5]:


jovian.commit(project='zerotoanalyst-numpy-array-operations')


# Let's begin by importing Numpy and listing out the functions covered in this notebook.

# In[6]:


import numpy as np


# ### Creating a NumPy array

# It is a normal Python list

# In[7]:


array = [1,2,6,5,3,3,6,78,98]
array


# Use `np.array()` to create a NumPy array

# In[8]:


np_array = np.array([2,3,5,6,7,76,5,4,32])
np_array


# Lets try to create a random array with specified size and shape

# In[9]:


random_array = np.random.randint(10, size=(5,3))
random_array


# In the random array above, it will generate a new random array every time you execute the cell.So, lets use `random.seed()` to generated a unique random array every time we execute.

# In[10]:


np.random.seed(3)
random_ar_1 = np.random.randint(20, size=(3,5))
random_ar_1


# In[11]:


np.random.seed(seed=0)
random_ar_2 = np.random.randint(20, size=(5,3))
random_ar_2


# In[12]:


np.random.seed(1)
random_ar_3 = np.random.randint(20, size=(2,3,4))
random_ar_3


# If we manually assign `seed=None` it will generate random arrays every time it executes

# ## Function 1 - Arithmetic Operation
# 
# ***Example 1 - Arithmetic operation*** are basic mathematical functions used in arrays, Here we can add, subtract, multiply and divide
# 

# In[13]:


a1 = np.array([2, 5, 6])
a1


# In[14]:


a2 = np.array([[3, 7, 9.6],
              [7, 4, 8.8]])
a2


# In[15]:


a1 + a2


# In[16]:


a1 - a2


# In[17]:


a3 = np.array([[4,7,9],
             [9,3,2.4]])
a3


# In[18]:


a2 * a1


# ***Example 2*** 
# - If we divide 2 arrays result will be a float value
# - If we check for modulas 

# In[19]:


a3 / a1


# In[20]:


a3 // a1


# In[21]:


a1 ** 3


# In[22]:


a2


# In[23]:


a2 % a1


# In[ ]:





# In[ ]:





# In[ ]:





# ## Funtcion 2 - Aggregation
# ***Example 1 -***  Aggregation = Performing same operation on a number of things 
# 

# In[24]:


a1


# In[25]:


sum(a1)


# In[26]:


np.sum(a1)


# In[27]:


a2


# In[28]:


np.mean(a2)


# In[29]:


np.max(a2)


# In[30]:


np.min(a2)


# ***Example 2*** 

# In[31]:


# Standard deviaiton = a measure of how spread out a group of numbers is from the mean 
np.std(a2)


# In[ ]:





# In[32]:


high_var_array = np.array([1, 10, 100, 200, 4000, 5000])
low_var_array = np.array([2,4,6,8,10,12])


# In[33]:


np.var(high_var_array), np.var(low_var_array)


# In[ ]:





# In[ ]:





# ***Example 3***

# In[ ]:





# In[34]:


# Variance = the measure of average degree to which each number is different to the mean
# Higher variance = higher range of numbers
# Lower variance = lower range of numbers

np.var(a2)


# In[35]:


np.std(high_var_array), np.std(low_var_array)


# In[36]:


np.mean(high_var_array), np.mean(low_var_array)


# ## Function 3 - Reshape & Transpose
# 
# 

# ***Example 1 - Reshape***

# In[37]:


a2


# In[38]:


a2.shape


# In[39]:


random_ar_3


# In[40]:


random_ar_3.shape


# In[41]:


a2_reshape = a2.reshape(2,3,1)
a2_reshape


# In[42]:


a2_reshape * random_ar_3


# ***Example 2 - Transpose***

# In[43]:


a2


# In[44]:


a2.shape


# In[45]:


a2.T


# In[46]:


a2.T.shape


# In[47]:


random_ar_3


# In[48]:


random_ar_3.shape


# In[49]:


random_ar_3.T


# In[50]:


random_ar_3.T.shape


# ## Function 4 - Linear Algebra

# In[51]:


np.random.seed(0)
mat1 = np.random.randint(10, size=(5,3))
mat2 = np.random.randint(10, size=(5,3))

mat1


# In[52]:


mat2


# In[53]:


# np.dot(mat1, mat2)


# In[54]:


mat3 = np.dot(mat1, mat2.T)
mat3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Function 5 - Comparision Operators

# In[55]:


a1


# In[56]:


a2


# In[57]:


a1 > a2


# In[58]:


a1 >= a2


# In[59]:


a1 == a2


# In[60]:


a1 != a2


# In[61]:


a1 > 5


# In[62]:


bool_array = a2 < 7
bool_array


# In[63]:


bool_array.dtype


# ## Function 6 - Sorting Operation

# ### Sort using `np.sort()`

# In[64]:


random_ar_1


# In[65]:


np.sort(random_ar_1)


# In[66]:


random_ar_2


# In[67]:


np.msort(random_ar_2)


# In[ ]:





# ### argsort `np.argsort()`

# In[68]:


sort_arr = np.array([9,53,16,387,125,678])
sort_arr


# In[69]:


np.argsort(sort_arr)


# In[70]:


np.argmax(sort_arr)


# In[71]:


np.argmin(sort_arr)


# In[ ]:





# In[72]:


random_ar_1


# In[73]:


np.argsort(random_ar_1)


# In[74]:


np.argsort(random_ar_1, axis=0)


# In[75]:


np.argsort(random_ar_1, axis=1)


# In[76]:


np.argmax(random_ar_1)


# In[77]:


np.argmax(random_ar_1, axis=0)


# In[78]:


np.argmax(random_ar_1, axis=1)


# In[ ]:





# In[79]:


jovian.commit()


# ## Conclusion
# 
# Summarize what was covered in this notebook, and where to go next

# ## Reference Links
# Provide links to your references and other interesting articles about Numpy arrays:
# * Numpy official tutorial : https://numpy.org/doc/stable/user/quickstart.html
# * ...

# In[83]:


jovian.commit()


# In[82]:


print('The End....')


# In[ ]:




