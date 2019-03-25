# Mod-a-Recipe

## Overview
Mod-a-Recipe is developed as my capstone project at Metis data science immersive bootcamp, to combine my passions for cooking and natural language processing. Using natural language and machine learning techniques, similar recipes are found given a selected recipe, and suggestions on modifications are provided to make a recipe your own.

## Data Source
Recipe data was scraped by [Eight Portions](https://eightportions.com/datasets/Recipes/) (~80,000 recipes used here) from [Epicurious](https://www.epicurious.com/),[AllRecipes](https://www.allrecipes.com/), and [FoodNetwork](https://www.foodnetwork.com/).

## Code at Github
Here is the github repo for this project: [link](https://github.com/pytgit/mod-a-recipe)

## Tools Used
Python is used for data acquisition, cleaning and modeling. Specific python libraries used include:
* Modeling: scikit-learn
* Natural language processing: Spacy

## Methodology Used
1. Data set from the three data sources (Epicurious, AllRecipes, Foodnetwork) are merged into one, and data cleaned to remove irrelevant information. (refer to code here: src/data/make_dataset.py)

2. Using a variety of machine learning tools (Python, scikit-learn, spacy), ~40,000 unique ingredients were extracted from the ingredients list of the recipes data, using NER modeling. (refer to code here: src/models/train_model-nlp.py and src/features/build_features.py)

3. Topic modeling technique (TFIDF word vector with non-negative matrix factorization) is then used to reduce dimensionality such that similar recipes can be calculated using cosine similarity. The NMF yielded 50 topics were seemed to be representative of certain types of recipes. For example:
   * Topic 1	(Asian recipes): soy sauce, sesame oil, green onion, ginger, sesame seed, rice vinegar, ginger root, scallion, rice wine vinegar, peanut oil
   * Topic 2	 (baking): unsalted butter, pure vanilla extract, whole milk, light brown sugar, fine salt, nutmeg, shallot, kosher salt and freshly ground pepper, extra-large egg, semisweet chocolate

   (refer to code here: src/models/train_model.py)

4. Similar recipes were found as a result, and differences in ingredients lists are highlighted as possible substitutions or enhancements to the selected recipe. Check out the application with sample results on [Mod-a-recipe](https://mod-a-recipe.herokuapp.com/)

## Resources
1. NYTimes Ingredients Phrase Tagger. [Github](https://github.com/NYTimes/ingredient-phrase-tagger)
2. Eight Portions (for recipes data). [Data set](https://eightportions.com/datasets/Recipes/)
