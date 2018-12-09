import json 
import pandas as pd
import numpy as np 
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD, SparsePCA , NMF, PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.svm import NuSVC, LinearSVC, SVC
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import recall_score,make_scorer, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score, average_precision_score
from sklearn.feature_extraction import text 
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from textblob import TextBlob
import datetime
from sklearn.metrics import cohen_kappa_score, fbeta_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import seaborn as sns 
import eli5
from matplotlib import rcParams






class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


#	Create a dataframe object consisting of one-hot encoded categorical features, ordinal features,  vectorized review data, 
# 	and the binary outcome of interest - whether a resteraunt failed inspections or not.
def clean_df(df,cat_feature_tags, ord_feature_tags, STOP_WORDS):

	#Vectorize reviews. 
	convert_text = DataFrameMapper([('relevant_reviews', 
		TfidfVectorizer(stop_words= STOP_WORDS, norm="l2", max_df=0.6, max_features=1000))],
		default=False)
	vect_text_array  = convert_text.fit_transform(df)
	vect_text_df = pd.DataFrame(vect_text_array)
	feature_tags = cat_feature_tags + ord_feature_tags
	feature_df = df[[col for feature in feature_tags for col in df.columns if feature in col]]
	feature_cols = [feature for feature in feature_df.columns]

	#	Create a Pandas dataframe object of categorical features and a tf-idf matrix of reviews. 
	X = pd.concat([feature_df, vect_text_df],axis=1)
	y = df['fail']
	text_cols = [col for col in X.columns if col not in feature_cols]
	
	return X,y,feature_cols,text_cols, feature_df


#	Split data into a training and test set with SMOTE option. The dataset is imbalanced: only about 20% of resteraunts fail inspections. Synthetic 		resampling (SMOTE) can be used to create a more balanced training set. But the danger is that the synthetic samples are too 'synthetic' and so I 		overfit the training data.  
def split_data_train_test (X,y,test_size,Use_SMOTE=True,k=4,alpha=0.7):
	
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size,shuffle=True)
	# Use SMOTE to oversample minority class in training data. To balance the cost of training on synthetic data with the benefit of more familiarity with the minority class, I apply a discounting factor of 0.8 to the resampling ratio that would fully equalize my two classes.  
	
	if Use_SMOTE == True:
		minority_count = y_train.sum()
		majority_count = len(y_train) - minority_count
		# We do not want LESS minority samples than the sample would produce. 
		if int(alpha * (majority_count//minority_count)) <= 1:
			res_ratio = 1
		else: 
			res_ratio = int(alpha * (majority_count//minority_count))
		X_train_SMOTE, y_train_SMOTE = SMOTE( ratio={0:majority_count, 1:int(res_ratio * minority_count)},k_neighbors=k,random_state=42).fit_sample(X_train, y_train)
		X_train_SMOTE = pd.DataFrame(X_train_SMOTE,columns=X.columns)
		return X_train_SMOTE, X_test ,y_train_SMOTE, y_test
	elif Use_SMOTE == False:
		return X_train, X_test, y_train, y_test

#Report metrics specific to imbalanced datasets and produce various plots. 
def test_classifier(clf_name,clf,X_train,X_test,y_train,y_test,threshold=0.3, alpha=None):
	results = []
	clf.fit(X_train, y_train)
	predicted_probas = clf.predict_proba(X_test)
	
	#Prioritize recall : lower threshold for predicting fail class 
	predictions = clf.predict(X_test)
	predictions = predicted_probas[:,1] > threshold

	print("Classification report for {}".format(clf_name))
	print(classification_report_imbalanced(y_test, predictions))
	try:
		print("Brier Score Loss")
		print(brier_score_loss(y_test,predicted_probas[:,1]))
		print("ROC AUC, Macro")
		print(roc_auc_score(y_test, predicted_probas[:,1], average='macro'))
	except Exception as e:
		print(e)
		print("Check to make sure probabilities are supported.")
	print("Mathew's Correlation")
	print(matthews_corrcoef(y_test, predictions))
	#Prioritize recall: Weight recall twice as much
	print("F Beta 2: Macro")
	print(fbeta_score(y_test, predictions, beta=2, average='macro'))
	print("F Beta 2: Fail Class")
	print(fbeta_score(y_test, predictions, beta=2, average='binary'))
	print("F Beta 2: Weighted")
	print(fbeta_score(y_test, predictions, beta=2, average='weighted'))


	scores = {"Brier Score": brier_score_loss(y_test,predicted_probas[:,1]),
	"Recall": recall_score(y_test, predictions), 
	"Precision":precision_score(y_test, predictions),
	"F": fbeta_score(y_test, predictions, beta=1, average='binary'),
	"F Weighted": fbeta_score(y_test, predictions, beta=1, average='weighted')}
	for metric in scores:
		results.append([clf_name,metric,scores[metric], alpha, threshold])

	# try:
	# 	Confusion matrix
	# 	sns.set_context("talk")
	# 	sns.set_palette("Set2")
	# 	ax = plt.subplot()
	# 	cm = confusion_matrix(y_test, predictions)
	# 	sns.heatmap(cm, annot=True, ax = ax)
	# 	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
	# 	ax.set_title('Confusion Matrix : {}'.format(clf_name)); 
	# 	ax.xaxis.set_ticklabels(['Pass', 'Fail']); ax.yaxis.set_ticklabels(['Pass', 'Fail']);
	# 	plt.show()
	# 	plt.close()


	# 	Calibration curves, PR curve, cumulative  gain curve, ROC
	# 	skplt.metrics.plot_roc_curve(y_test, predicted_probas,title=clf_name)
	# 	skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas,title=clf_name)
	# 	skplt.metrics.plot_cumulative_gain(y_test, predicted_probas,title=clf_name)
	# 	skplt.metrics.plot_calibration_curve(y_test,[predicted_probas],[clf_name], title = "{} Calibrated Probabilities".format(clf_name))
	# 	plt.show()
	# 	plt.close()
	# except Exception as e:
	# 	print(e)


	return results 


# Build ML models 
def build_models(X,y,feature_cols,text_cols):

	svc_txt = Pipeline([('col_selector', 
		ItemSelector(key=[col for col in X.columns if col in text_cols])),
		('reduce_dims', PCA(n_components=80,  random_state=42)),
	    ('stdize_feats', StandardScaler()), 
	    ('calib_probs', CalibratedClassifierCV
		(cv=5, method='sigmoid', base_estimator = 
		SVC(
			class_weight='balanced',
			probability=True,
			max_iter=3000,
			kernel='sigmoid',
			C = 18,
			gamma = 0.3
			)))])

	lr_txt = Pipeline([('col_selector', 
		ItemSelector(key=[col for col in X.columns if col in text_cols])),
		('reduce_dims', PCA(n_components=80,  random_state=42)),
	    ('stdize_feats', StandardScaler()), 
	    ('calib_probs', CalibratedClassifierCV
		(cv=5, method='isotonic', base_estimator = 
	    LogisticRegression(
	    	random_state=42,
			warm_start=True,
			C = 10,
			class_weight='balanced',
			solver="newton-cg",	
			penalty="l2",
			)))])


	lr_feat = Pipeline([('col_selector', 
		ItemSelector(key=[col for col in X.columns if col in feature_cols])),
		('reduce_dims', PCA(n_components=95)),
	    ('stdize_feats', StandardScaler()), 
	    ('calib_probs', CalibratedClassifierCV
		(cv=5, method='isotonic', base_estimator = 
	    LogisticRegression(
	    	class_weight='balanced',
			C = 0.06,
			warm_start=True,
			random_state=42,
			)))])


	rf_feat = Pipeline([('col_selector', 
		ItemSelector([col for col in X.columns if col in feature_cols])),
		('stdize_feats', StandardScaler()), 
		('calib_probs', CalibratedClassifierCV
		(cv=5, method='isotonic', base_estimator = 
			RandomForestClassifier(
				class_weight='balanced',
				random_state=42,
				max_features="sqrt",
				max_depth= 10,
				n_estimators=1000
			)))])

	
	base_estimators = {"feat_clfs":[('rf_feat', rf_feat), ('lr_feat', lr_feat)],
	"txt_clfs": [('rf_txt', lr_txt), ('svc_txt', svc_txt)]}
	txt_voter = VotingClassifier(estimators=base_estimators['txt_clfs'],voting="soft")
	feat_voter = VotingClassifier(estimators=base_estimators['feat_clfs'],voting="soft")
	master_voter = VotingClassifier(estimators=[('txt_voter', txt_voter), ('feat_voter', feat_voter)], voting='soft', weights=[2,1])

	model_list = [
		('Feature Voter', feat_voter),
		('Text Voter', txt_voter),
		('Master Voter', master_voter),
		('Support Vector Machine (Text)', svc_txt), 
		('Logistic Regression (Text)', lr_txt),
		('Random Forest (Features)', rf_feat), 
		('Logistic Regression (Features)', lr_feat)]




	results = []
	for trial in range(10):
		for (name, model) in [('Master Voter', master_voter)]
			for alpha in [0.7,0.8,0.9,1]:
				for threshold in [0.3,0.4,0.5]:
					X_train, X_test, y_train, y_test = split_data_train_test(X,y,test_size=0.40,alpha=alpha)
					scores = test_classifier(name, model, X_train, X_test,y_train,y_test,alpha=alpha,threshold=threshold)
					for score in scores: 
						results.append(score)
						print(score)
	df = pd.DataFrame(results, columns = ['model', 'metric', 'score', 'alpha', 'threshold'])
	df.to_csv('alpha_threshold_brier.csv', encoding='utf-8', index=False)
	print(df)

	results = []
	for trial in range(20):
		for (name, model) in model_list:
			X_train, X_test, y_train, y_test = split_data_train_test(X,y,test_size=0.40)
			scores = test_classifier(name, model, X_train, X_test,y_train,y_test)
			for score in scores: 
				results.append(score)
	df = pd.DataFrame(results, columns = ['model', 'metric', 'score', 'alpha', 'threshold'])
	print(df)


def plot_metrics(brier_alpha, compare_clfs):
	sns.set_context("talk")
	rcParams.update({'figure.autolayout': True})
	#Alpha vs brier score 
	a = pd.read_csv('alpha_threshold_brier.csv')
	brier_df = a.loc[a['metric'] == 'Brier Score']
	d = sns.lmplot(x="alpha", y="score", order=3, x_jitter=0.01, palette=  "Set3", data=brier_df)
	plt.title("Alpha vs Brier Score \n (Master Voter)")
	plt.ylabel("Brier Score \n (Lower = Better)")
	plt.xlabel("Alpha")
	plt.show()
	

	

# Convert the list of firm dicts to long format (one entry per inspection) and flatten the dictionaries. 
# After flattening, use one-hot encoding on feategorical features. 
def Flatten_WideToLong(matched_firms,feature_tags):
	reviewed_firms_long = []
	for firm in matched_firms:
		for inspection in firm['inspection_data']:
			temp_dict = { 
				"relevant_reviews":inspection['relevant_reviews'],
				"reviews_sentiment":TextBlob(inspection['relevant_reviews']).sentiment.polarity,
				"business_id":firm["business_id"], 
				"inspection.InsNum":inspection['InsNum'], 
				"fail":inspection['fail'], 
				"stars":int(firm['stars']),
				"review_count":int(firm['review_count']), 
				"postal_code":firm['postal_code'],
				"neighborhood":firm['neighborhood'], 
				"attributes":firm['attributes'] 
				}

			dt = datetime.datetime.strptime(inspection['date'], "%Y-%m-%d %H:%M:%S")
			for (label,value) in [("date.month",dt.month), ("date.year",dt.year)]:
				temp_dict.update({label:value})
			
			try:
				temp_dict.update({"price":firm['attributes']['ResterauntsPriceRange2']})
			except:
				temp_dict.update({"price":2})

			for category in firm['categories']:
				try:
					temp_dict.update({"categories.{}".format(category):1})
				except:
					temp_dict.update({"categories.{}".format(category):0})
			
			reviewed_firms_long.append(temp_dict)

	df = (json_normalize(reviewed_firms_long)).fillna(0)
	for (to_replace,replacement) in [(True,1), (False,0), ("Yes",1), ("yes", 1), ("no", 0), ("No", 0)]:
		df.replace(to_replace=to_replace, value=replacement, inplace=True)
	
	#Use one hot encoding on all feature columns of 'object' type (ie: those containing categorical data). 
	feature_cols = [column for tag in feature_tags for column in df.columns if tag in column]
	categorical_cols = [col for col in df.select_dtypes(include=['object'])]
	cols_to_convert = set(feature_cols).intersection(categorical_cols)
	df_dummied = pd.get_dummies(df,columns=cols_to_convert,dummy_na=False, sparse=True)
	return df, df_dummied


# Explain weights of classifiers 
def explain_classifiers(X,y, df, feature_cols, stop_words,n=20):

	lr_txt = LogisticRegression(
			random_state=42,
			warm_start=True,
			C = 10,
			class_weight='balanced',
			solver="newton-cg",	
			penalty="l2",
		)
	rf = RandomForestClassifier(
				random_state=42,
				max_features="sqrt",
				max_depth= 10,
				n_estimators=1000 
	)
	lr_cat = LogisticRegression(class_weight='balanced',
			C = 0.06,
			warm_start=True,
			random_state=42)

	X_text = df['relevant_reviews']
	vect = TfidfVectorizer(stop_words=stop_words, norm="l2",  max_df=0.6, max_features=1000)
	X_vect = vect.fit_transform(X_text)
	feature_df = df[[col for col in X if col in feature_cols]]
	X_feat = pd.concat([feature_df],axis=1)

	#LR cat Feature Importance 
	rcParams.update({'figure.autolayout': True})
	lr_cat.fit(X_feat, y)
	weights = list(zip(lr_cat.coef_[0], X_feat.columns))
	weights.sort(reverse=True)
	weights_df = pd.DataFrame(weights[:20], columns=['weight', 'feature'])
	sns.set_context("talk")
	c = sns.barplot(x="weight", y='feature', data=weights_df, palette="Set3")
	plt.title("Yelp Resteraunt Features Associated With Resteraunt Inspection Failures \n (Positive weights imply increased risk of failing.)")
	plt.xlabel("Logistic Regression Weights")
	plt.ylabel("Feature")
	plt.xticks(rotation=90)
	plt.show()

	#RF Feature Importance 
	rcParams.update({'figure.autolayout': True})
	rf.fit(X_feat, y)
	weights = list(zip(rf.feature_importances_, X_feat.columns))
	weights.sort(reverse=True)
	weights_df = pd.DataFrame(weights[:20], columns=['weight', 'feature'])
	sns.set_context("talk")
	c = sns.barplot(x="weight", y='feature', data=weights_df, palette="Set3")
	plt.title("Yelp Resteraunt Features Associated With Resteraunt Inspection Failures \n (Larger weights imply increased importance)")
	plt.xticks(rotation=90)
	plt.xlabel("Random Forest Weights")
	plt.ylabel("Feature")
	plt.show()


	rcParams.update({'figure.autolayout': False})
	lr_txt.fit(X_vect,y)
	weights_df = eli5.explain_weights_df(lr_txt, vec=vect, top=20,target_names=y)
	sns.set_context("talk")
	b = sns.barplot(x="feature", y='weight', data=weights_df, palette="Set3")
	plt.xlabel('Word')
	plt.title("Yelp Resteraunt Features Associated With Resteraunt Inspection Failures \n (Positive weights imply increased risk of failing.)")
	plt.ylabel("Logistic Regression Weights")
	plt.xticks(rotation=45)
	plt.show()




def main(): 
	custom_stops = ("nan", "NaN")
	STOP_WORDS = text.ENGLISH_STOP_WORDS.union(custom_stops)
	cat_feature_tags = ["attributes", "categories","neighborhood"]
	ord_feature_tags = ['price', 'stars', 'date', 'inspection.InsNum']
	with open("matched_firms.json") as f:
		matched_firms = f.read().strip().split("\n")
		matched_firms_matrix = [json.loads(firm) for firm in matched_firms]
	print("Flattening data...")
	flat_df,flat_df_dummied = Flatten_WideToLong(matched_firms_matrix, cat_feature_tags)
	print("Cleaning data...")
	X,y,feature_cols,text_cols, feature_df = clean_df(flat_df_dummied, cat_feature_tags, ord_feature_tags, STOP_WORDS)
	print("Implemening ML Models...")
	models = build_models(X,y,feature_cols,text_cols)
	explain_classifiers(X,y, flat_df_dummied, feature_cols,STOP_WORDS)
	plot_alpha_metrics("alpha_threshold_brier.csv")


	



	

	



main()