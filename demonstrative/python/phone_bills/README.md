# Phone Call Classification

I got really tired of going through my phone bill every month to figure out which calls I should expense due to being business-related (I don't have a separate line for work) and decided to try to solve the problem with a classifier.

This project contains an IPython notebook detailing the process and some of the results I found.  Long story short, it's easy to do that sort of thing with a computer and I'm happy to say I'll never need to do it manually again!

Here is a copy of that notebook with lots of details:
[Phone Call Classification Project](https://rawgit.com/eric-czech/portfolio/master/demonstrative/python/phone_bills/phone_bills.html).

There are plenty of findings and in there, but one of the most interesting IMO was the relative influence of each feature in the model:
<img src="https://rawgit.com/eric-czech/portfolio/master/demonstrative/python/phone_bills/feature_importance.png"/>

The most important input to determining whether or not any one phone call was business-related was the phone number involved in the call.  That's pretty obvious, but its limiting in that the numbers I receive calls from each month tend to change frequently so in order for the classification to generalize well, other features like area code, call length, and call time are needed.

Anyways, it's a cool approach and I suspect it would work well for a lot of people.
