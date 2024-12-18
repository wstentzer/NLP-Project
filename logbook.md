# 02466 Project work in Artificial Intelligence and Data LOGBOOK

- William Krøyer Stentzer, s214645@student.dtu.dk
- Mads Bjørke Vejbæk, s225783@student.dtu.dk

# Project Meetings

## Week 2 10-09-24:
Walked through the research paper we recived from our supervisor, in preperation for the next week.

## Week 3 17-09-24:
We also skipped out meetings as the we other assigment that were urgent.

## Week 4 24-09-24:
Talked further about the research paper we reciveed from our supervisor. And also som we found ourselves, but unfortunately they weren't as relevant as we hoped.

## Week 5 01-10-24:
We started on the projekt plan (and everything that includes). We need to finish project canvas, create a garnt chart and read through the introduction and data description to ensure its understandable. We also wrote the coorporation agreement, which we need to sign before sunday. 

## Week 6 08-10-24:

Started writing the code in a jupyter notebook.

## Week 7 15-10-24:

One of the groups memebers has fallen ill so metting is postponed to next week

## Week 8 22-10-24:

Caught up all group members on the Supervisor meeting last week. And begun further work on the code implementaion

## Week 9 29-10-24:

We kept working on the implementation. As the most important thing right now is getting a working prototype.

## Week 10 05-11-24:

From the last Supervisor meeting we found some strange behaivior with regards to the accuracy of the ensemble.
Which we will insvestigate and hopefully fix.

## Week 11 12-11-24:

Implement the metrics and histograms, that were suggested from our supervisor, to get a better understanding of out models perfomance 


# Supervisor Meetings

## Week 1 05-09-24:
We met for the first time with the supervisor and talked about the possible project we can work with him on.

## Week 2 12-09-24: 
We talked about the project and recieved a research paper (DUDES: Deep Uncertainty Distillation using Ensembles for Semantic Segmentation) that we should read.

## Week 3 19-09-24:
Supervisor was at a conference so the meeting was postponned to the next week

## Week 4 26-09-24:
Talked more indepth about the project. About the architecture and how our project differs from the paper we recived last week. And agreed upon the date for the first deliverable (Project plan etc.)

## Week 5 03-10-24:
### Questions:
- Method walktrough

short description of llms
distilation
evaluation metrics
ensemble consists of mlp

- Project title? suggestion: Distillation of Deep Ensemble for Uncertainty Quantification for Semantic Classification

Yes
- Code base are we starting from scratch?

Yes, but most of the first part of the project should have code that could be used on the web
- Approval of the research questions

Rephrase the questions and send them to michael to get approvel
- Guidance on how to better write a logbook

Expand the description of the meetings relating the decesions and considerations. Make it more clear what we got out of the meeting

- Are we sending the first handin to both Michael and Morten?

Yes
- In the context of research question 3 how does out-of-distribution relate to text classification.

Find a different dataset maybe in danish and test the models on that

### Next steps:
Decide on a LLM and train a MLP

New meeting time: Wednsday 10.15 from the 16 october

## Week 6 10-10-24:

Find some other papers on Ensemble to validate that our accuracy is within a reasonable range of that

## Week 7 17-10-24:

No notes. Just keep working on the implementation of the Ensemble

## Week 8 23-10-24:

Plot accuracy for the MLPs and the ensemble with size 1 to N

## Week 9 30-10-24:

It looks like we have a data leak in the training and validation. So we need to investigate that and fix it for next week

## Week 10 06-11-24:

Create histograms over the following metrics to further validate the competence of the ensemble 
- Accuracy
- Negative Log Likelihood
- Expected Calibration Error
