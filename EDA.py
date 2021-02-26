import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocess import df_blog

#%% Age of users
px.histogram(data_frame= df_blog, x= 'age').show()

#%% Blog posts number in time
df_blog['date_quarters'] =  df_blog.date_dformat.apply(lambda x: x.to_period('Q'))

#%%
df_quarterly_grouped = df_blog.groupby(['date_quarters']).count()
px.bar(x = df_quarterly_grouped.index.astype(str), y = df_quarterly_grouped.id,
       labels = {'x': 'Quarter','y':'Tweets number'}).show()

#%% Topic balance
px.bar(df_blog.topic.value_counts(), labels= {'value':'number of posts', 'index': 'topic'},
       title= 'Topic posts cardinality').show()

#%% Groupby topic
df_topic_grouped = df_blog.groupby(['topic']).median()

px.bar(x = df_topic_grouped.index, y = df_topic_grouped.age,
       labels = {'x': 'topic','y':'median user age'}).show()

#%% Men and women

df_gender_grouped = df_blog.groupby('gender').mean()

print(f"Average number of words in text: \n Men: {df_gender_grouped.loc['male', 'word_number']:.2f}"
      f" \n Women: {df_gender_grouped.loc['female', 'word_number']:.2f}")

#%%

df_men = df_blog[df_blog.gender == 'male']
df_women = df_blog[df_blog.gender == 'female']
df_men_topic = df_men.groupby(['topic']).count()
df_women_topic = df_women.groupby(['topic']).count()

fig_gender_topic = go.Figure()
fig_gender_topic.add_trace(go.Bar(x = df_men_topic.index, y = df_men_topic.id, name = 'Male'))
fig_gender_topic.add_trace(go.Bar(x = df_women_topic.index, y = df_women_topic.id, name = 'Female'))
fig_gender_topic.update_layout(title= 'Posts topic cardinality considering gender')
fig_gender_topic.show()

#%% Check Benford's law

df_blog['first_digit'] = df_blog['word_number'].astype(str).apply(lambda x: x[0])
df_benford = pd.DataFrame([0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046], columns=['Frequency'])
df_benford.index +=1
first_digit = df_blog['first_digit'].value_counts(normalize = True)

fig_benford = go.Figure()
fig_benford.add_trace(go.Scatter(x = first_digit.index, y = first_digit.values, name = 'Posty z blogów'))
fig_benford.add_trace(go.Scatter(x = df_benford.index, y= df_benford['Frequency'], name = 'Norma'))
fig_benford.update_layout(title = "Check Benford's law in number of words per post")
fig_benford.show()

from sklearn.metrics import mean_absolute_error

mean_absolute_error(df_benford['Frequency'], first_digit)