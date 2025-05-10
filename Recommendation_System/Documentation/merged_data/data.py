import numpy as np
import pandas as pd


jobs = pd.read_csv('job_skills.csv')

cols = []
for col in jobs.columns:
    cols.append(col)

jobs['Description'] = " "

for col in cols:
    jobs['Description'] += " " + jobs[col]

print(jobs['Description'][0])

jobs.to_csv('jobs_skills_with_description.csv', index=False)

resumes = pd.read_csv('resumes_indeed_com-job_sample.csv')

cols = []
for col in resumes.columns:
    cols.append(col)

resumes['Description'] = " "

for col in cols:
    resumes['Description'] += " " + resumes[col]

print(resumes['Description'][0])

resumes.to_csv('resumes_indeed_com-job_sample_with_description.csv', index=False)