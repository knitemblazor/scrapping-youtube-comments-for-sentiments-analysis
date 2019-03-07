# Youtube  scrapped comments and sentiment analysis on them

Here scrapped comments are labelled by checking for occurance of positive and negative word occurances. instead one could also use Sentiwordnet for proper labelling
 As one can see the performance is quite poor.
the way to get around with it is by doing more preprocessing and obtaining better labelled data

### complete_nlp_handling_food_comments_dataset.ipynb
i have used a multi layer perceptron as a classifier and includes the crawing code in one of the cells .It includes all the preprocessing steps involved

### tokenizer[loss 70%,acc69%conv27%].ipynb
  
Is a python notebook that has the info on implementing a cnn and an mlp and the accuracy obtained

users are requested to run the script on google colab or with a pc with  a min of 8Gb RAM
