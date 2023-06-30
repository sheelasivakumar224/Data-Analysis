#!/usr/bin/env python
# coding: utf-8

# # Data Visualisation 

# ## Matplotlib and Seaborn
# 
# 1. Installing the Libraries
# 
# !pip install matplotlib
# !pip install seaborn
# 
# 2. Importing the Libraries 
# 
# import matplotlib.pyplot as plt
# %matplotlib inline (this line embeds the graph within the Jupyter notebook and not on a popup window)
# 
# import seaborn as sns

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Line Chart

# In[2]:


# Create an array
scores = [12,22,24,46]


# In[3]:


plt.plot(scores)


# #### Note:
# To avoid the "Unintendend output: [<matplotlib.lines.Line2D at 0x2322e952940>]" we use ";" at the end of the statement of the entire cell 

# In[4]:


plt.plot(scores);


# #### Adding More Info to the Graph
# 
# Adding a Meaningful X-axis using an another array
# Labelling the x-axis and y-axis 

# In[8]:


scores = [12,22,24,46]
sub = ['sub_1','sub_2','sub3','sub_4']
plt.plot(sub,scores);  #plot(x_axis,y_axis)


# In[12]:


plt.plot(sub,scores)
plt.xlabel("Subject")
plt.ylabel("Mark");


# #### Note
# A plt inside a single cell will plot only one graph 

# #### Plotting a Multiple Lines 

# In[13]:


student_1 = [12,22,24,46]
student_2 = [34,2,4,23]
plt.plot(sub,student_1)
plt.plot(sub,student_2)

# Labelling X-axis and Y-axis
plt.xlabel("Subject")
plt.ylabel("Scores")

# Setting Title of the graph
plt.title("Mark of PT");


# In[16]:


student_1 = [12,22,24,46]
student_2 = [34,2,4,23]

# Setting up the marker 
plt.plot(sub,student_1,marker = 'o')
plt.plot(sub,student_2,marker = 'x')

# Labelling X-axis and Y-axis
plt.xlabel("Subject")
plt.ylabel("Scores")

# Setting Title of the graph
plt.title("Mark of PT")

# Setting up a identifier : Legend
plt.legend(['student_1','student_2']);


# In[17]:


student_1 = [12,22,24,46]
student_2 = [34,2,4,23]

# Setting up the marker 
plt.plot(sub,student_1,marker = '*')
plt.plot(sub,student_2,marker = 'D')

# Labelling X-axis and Y-axis
plt.xlabel("Subject")
plt.ylabel("Scores")

# Setting Title of the graph
plt.title("Mark of PT")

# Setting up a identifier : Legend
plt.legend(['student_1','student_2']);


# #### Styling of Lines
# 
#  - color or c : color of the line 
#  - linestyle or ls : example dotted/continuoue
#  - linewidth or lw : width of the line
#  - markersize or ms : size of the marker
#  - markeredgecolor or mec : color of the outline of the marker 
#  - markeredgewidth or mew : width of the outline of the marker
#  - markerfacecolor or mfc : Color of the marker inner fill
#  - alpha : opacity of the plot 
# 
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# 

# In[18]:


student_1 = [12,22,24,46]
student_2 = [34,2,4,23]
sub = [1,2,3,4]


# In[23]:


plt.plot(sub,student_1,marker='X',ls=':',c='c',lw=2)
plt.plot(sub,student_2,marker='D',ls='--',c='m',lw=3)
plt.title("PT mark")
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.legend(['Student 1','Student 2']);


# ##### Shorthand for linestyle, marker, linecolor
# 
# fmt = '[marker][linestyle][linecolor]'
# example 's-b'

# In[24]:


plt.plot(sub,student_1,'s-c')
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.title("PT mark");


# ##### Note
# What if? line style is not given 
# The graph will be plotted with only the markers 

# In[25]:


plt.plot(sub,student_1,'Dm');


# In[26]:


sns.set_style("white")


# In[27]:


plt.plot(sub,student_1);


# In[37]:


sns.set_style("ticks")
plt.plot(sub,student_1);


# #### Note:
#  The difference between the white and ticks is that we can see that each points in the axis is marked by a small-line(tick) in ticks style 
#  
#  ![image.png](attachment:image.png)  ![image-2.png](attachment:image-2.png)

# In[30]:


sns.set_style("dark")


# In[31]:


plt.plot(sub,student_1);


# In[33]:


sns.set_style("whitegrid")


# In[34]:


plt.plot(sub,student_1);


# In[35]:


sns.set_style("darkgrid")
plt.plot(sub,student_1);


# #### Changing Figure size
# 
# plt.figure(figsize = (length , breadth))

# In[39]:


plt.figure(figsize = (10,4))
plt.plot(sub,student_1);


# In[41]:


plt.figure(figsize = (4,8))
plt.plot(sub,student_1);


# In[49]:


plt.plot(sub,student_1)
plt.figure(figsize = (4,8));


# #### Note:
#    Figsize should be defined first 
#    Then the graph definition must be given

# In[43]:


import matplotlib
matplotlib.rcParams


# In[44]:


matplotlib.rcParams['font.fantasy'] = 'Chicago'


# In[47]:


plt.plot(sub,student_1)
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.title("PT marks");


# ## Scatterplot

# In[50]:


import pandas as pd


# In[52]:


# Loading the data into dataframe
df = pd.read_csv("IRIS.csv")


# In[53]:


df.head()


# In[54]:


df.info()


# In[55]:


df['species'].unique()


# In[57]:


plt.plot(df['sepal_length'],df['sepal_width'])
plt.xlabel('Sepal length')
plt.ylabel("Sepal width");


# In[58]:


plt.plot(df.sepal_length,df.sepal_width);


# In[61]:


sns.set_style("darkgrid")
plt.plot(df.sepal_length,df.sepal_width,'ob')
plt.xlabel('Sepal length')
plt.ylabel("Sepal width");


# In[65]:


sns.scatterplot


# In[72]:


sns.scatterplot(data = df,x="sepal_length",y="sepal_width");


# In[75]:


sns.scatterplot(data = df , x = 'sepal_length',y = 'sepal_width', hue = 'species');


# In[85]:


plt.figure(figsize = (8,5))
sns.scatterplot(data = df , x = 'sepal_length',y = 'sepal_width',hue='species',s=60);


# In[87]:


plt.figure(figsize = (8,5))
sns.scatterplot(data = df , x = 'sepal_length',y = 'sepal_width',hue='species',size = 'species',sizes=(20, 200));


# ### Scatterplot vs Lineplot
#  "unique markers" or "colors" to different categories.

# ## Histogram

# In[88]:


plt.hist


# In[91]:


df.describe()


# In[92]:


plt.hist(df.sepal_width);


# In[94]:


plt.hist(df['sepal_width'],bins=8)


# In[96]:


plt.hist(df['sepal_width'],bins=[1,2.5,4,5]);


# In[98]:


import numpy as np
plt.hist(df.sepal_width,bins=np.arange(2,5,0.25));


# ## Multiple Histogram

# In[101]:


setosa_df = df[df['species']=='Iris-setosa']


# In[102]:


setosa_df.head()


# In[104]:


versicolor_df = df[df['species']=='Iris-versicolor']
versicolor_df.head()


# In[105]:


virginica_df = df[df['species']=="Iris-virginica"]
virginica_df.head()


# In[107]:


plt.hist(setosa_df.sepal_width,alpha=0.4)
plt.hist(virginica_df.sepal_width,alpha = 0.4);


# In[110]:


plt.title("Distribution of Sepal width")
plt.hist([setosa_df.sepal_width,versicolor_df.sepal_width,virginica_df.sepal_width],stacked = True)
plt.legend(['Setosa','Versicolor','Virginica']);


# ## Bar Chart

# In[113]:


years = [2002,2003,2004,2005,2006,2007]
apples = [3,6,9,5,8,4]
oranges = [4,8,9,7,6,8]


# In[116]:


plt.plot(years,apples);


# In[117]:


plt.bar(years,apples);


# In[122]:


plt.bar(years,apples)
plt.plot(years,apples,c='m',marker='D');


# In[125]:


plt.bar(years,apples)
plt.bar(years,oranges,bottom=apples);


# #### Barchart vs histogram
# 
# Barplots : Discrete|Uniform|Equal width of bar|height represent frequency count|comparisons between different categories or groups. 
# histogram : Continuous|Non-uniform| unequal width of bar based on the bins range|height represent frequency density|frequency or count of values within specific intervals or bins.

# ### Bar plots with Averages 

# In[126]:


tips_df = sns.load_dataset("tips");
tips_df.head()


# In[132]:


sns.barplot


# In[133]:


bill_avg_df = tips_df.groupby('day')[['total_bill']].mean()
bill_avg_df.head()


# In[135]:


plt.bar(bill_avg_df.index,bill_avg_df['total_bill']);


# In[131]:


sns.barplot(data = tips_df,x="day",y="total_bill");


# In[136]:


sns.barplot(data=tips_df,x='day',y='total_bill',hue='sex')


# In[138]:


sns.barplot(data = tips_df,x='total_bill',y='day')


# ## HeatMap

# In[ ]:





# ## Images

# In[144]:


from PIL import Image


# In[145]:


# Loading the Image 
img = Image.open("Hello_K.jpg")


# In[147]:


# Displaying the Image
plt.imshow(img);


# In[148]:


img_array = np.array(img)


# In[149]:


img_array.shape


# In[150]:


img_array


# In[151]:


plt.grid(False)
plt.axis("off")
plt.imshow(img)
plt.title("Follow your Heart");


# In[155]:


plt.imshow(img_array[300:400,100:300])
plt.grid(False)
plt.axis('off');


# In[176]:


fig,axes = plt.subplots(2,3,figsize=(15,8))
plt.tight_layout(pad = 3)

# Line plot
axes[0,0].set_title("Line Plot")
axes[0,0].set_xlabel("Subject")
axes[0,0].set_ylabel("Scores")
axes[0,0].plot(sub,student_1,'D:m')
axes[0,0].plot(sub,student_2,'s-c')
axes[0,0].legend(['Student 1','Student 2']);

# Scatter plot
axes[0,1].set_title("Scatter Plot")
axes[0,1].set_xlabel("Sepal length")
axes[0,1].set_ylabel("Sepal width")
sns.scatterplot(data = df,x= 'sepal_length',y='sepal_width',hue='species',ax=axes[0,1]);

# Histogram
axes[0,2].set_title("Histogram")
axes[0,2].set_xlabel("Sepal_width")
axes[0,2].set_ylabel("Count")
axes[0,2].hist(df.sepal_width,bins=10);

# Barplot
axes[1,0].set_title("Bar Plot")
axes[1,0].set_xlabel("day")
axes[1,0].set_ylabel("total_bill")
sns.barplot(data = tips_df,x ="day",y = "total_bill",hue="sex",ax=axes[1,0]);

# Heatmap
axes[1,1].set_title("HeatMap")
axes[1,1].set_xlabel("Subject")
axes[1,1].set_ylabel("Scores")
sns.heatmap([sub,student_1],ax=axes[1,1])

# Image
axes[1,2].set_title("Image")
axes[1,2].imshow(img)
axes[1,2].grid(False)
axes[1,2].axis("off");

