# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    doc = doc.lower()
    if keep_internal_punct == False:
    	results = re.sub('\W+',' ',doc.lower()).split()
    else:
    	#results = re.sub('(\Z\W+)|(\A\W+)',' ',doc.lower()).split()
      #results = re.sub('\Z\W+', ' ', doc.lower()).split()
      
      results = doc.split()
      for it in range(0, len(results)):
        results[it] = re.sub('\A\W+', '', results[it])
        results[it] = re.sub('\W+\Z', '', results[it])
        #print(results[it])
      
    #print('tokenize pass')
    return np.array(results)
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    #tokens = list(tokens)
    #tokens = sorted(tokens)
    #print(tokens)
    for item in tokens:
      string = 'token=' + item
      if string in feats:
        feats[string] += 1
      else:
        feats[string] = 1
    #print('token_features pass')
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    length = len(tokens)
    for index in range(0, length-k+1):
      iterate = k
      while(iterate >1):
        new_index = index + k-iterate
        for i in range(1, iterate):
          new_key = "token_pair=" + tokens[new_index] + "__" + tokens[new_index + i]
          if new_key in feats:
            feats[new_key] += 1
          else:
            feats[new_key] = 1
        iterate -= 1
    #print('token_pair pass')
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats["neg_words"] = 0
    feats["pos_words"] = 0
    for items in tokens:
      items = items.lower()
      if items in neg_words:
        feats["neg_words"] += 1
      if items in pos_words:
        feats["pos_words"] += 1
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    for fns in feature_fns:
      fns(tokens, feats)
    featurize_result = feats.items()
    featurize_result = sorted(featurize_result)
    return featurize_result
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    result = []
    indptr = [0]
    vocab = {}
    for items in tokens_list:
      result.append(featurize(items, feature_fns))
    #print('vec step1')
    #delete features according to minfreq.
    #construct vocab
    for rr in result:
      for ee in rr:
        if ee[0] not in vocab:
          vocab[ee[0]] = 1
        else:
          vocab[ee[0]] += 1
    for kk, vv in list(vocab.items()):
      if vv < min_freq:
        del vocab[kk]
    #print('vec step2')
    #sort the key apl
    keys = list(vocab.keys())
    keys = sorted(keys)
    x=0
    for ke in keys:
      vocab[ke] = x
      x = x+1
    #construct csr_matrix
    #r index of row
    #c index if col
    #i index of items in result, j index if items in vocab.
    ##############################use dict here to store list to a dict#############
    #print('vec step3')
    data = []
    row = []
    col = []
    for r in range(0, len(result)):
      dic = {}
      for k,v in result[r]:
        dic[k] = v
      for key in keys:
        if key in dic:
          data.append(dic[key])
          row.append(r)
          col.append(vocab[key])
            
    datas = np.array(data,dtype='int64')
    rows = np.array(row,dtype='int64')
    cols = np.array(col,dtype='int64')
    l = len(list(vocab.keys()))
    X = csr_matrix((datas,(rows, cols)),shape = (len(result), l))
    return (X, vocab)

    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
      clf.fit(X[train_ind], labels[train_ind])
      predictions = clf.predict(X[test_ind])
      accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    #get combinations of feature functions
    function_cb = []
    result = []
    for num in range(1,len(feature_fns) +1):
      function_cb.extend(combinations(feature_fns,num))
    #print(function_cb)
    #tokenize, punct_vals may be True of False
    tokens_list = []
    #print(type(labels))
    #print(labels)
    #for items in punct_vals:
    #  tokens_list.append(tokenize(docs,items))
    #vectorize, different fuction combinations, different tokenlist, different min_freqs.
    for items in punct_vals:
      tokens_list = [tokenize(d,items) for d in docs]
      for mf in min_freqs:
        for functions in function_cb:
          X,vocab = vectorize(tokens_list,functions,mf)
          clf = LogisticRegression()
          accuracy = cross_validation_accuracy(clf,X,labels,5)
          temp = {}
          temp['punct'] = items
          temp['features'] = functions
          temp['min_freq'] = mf
          temp['accuracy'] = accuracy
          result.append(temp)

    result = sorted(result, key =lambda x: x['accuracy'], reverse = True)
    #print(result)
    return result
    pass


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    results = sorted(results,key = lambda x:x['accuracy'])
    plt.plot([re['accuracy'] for re in results])
    plt.xlabel('setting')
    plt.ylabel('accuracies')
    plt.savefig('accuracies.png')
    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    #for each kind of setting--> features/min_freq....
    #scan each element in result, if the setting name which is result[i][keys] is the same, then add to accuracy array.
    #settings is a dict of list, which key is the name of setting, and list of accuracies
    settings = {}
    for keys in results[0]:
      if keys != 'accuracy':
        for element in results:
          #print(type(keys))
          #print(type(element[keys]))
          new_key = ''
          if keys == 'features':
            new_key = keys + '='
            #print(type(element[keys]))
            for func in element[keys]:
              func = str(func)
              fs = func.split()
              new_key += fs[1]
              new_key +=" "
          else:
            new_key = keys +"="+ str(element[keys])
          if new_key in settings:
            settings[new_key].append(element['accuracy'])
          else:
            settings[new_key]= []
            settings[new_key].append(element['accuracy'])
    #store setting to a list of tuples and sorted.
    result = []
    for items in settings:
      acc = np.mean(settings[items])
      sett = items
      result.append((acc,sett))
    return sorted(result,reverse= True)
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    #tokenize the docs. 
    #vectorize
    #fit logistic clf( need rewrite)
    punct = best_result['punct']
    functions = best_result['features']
    min_freq = best_result['min_freq']
    
    tokens_list = [tokenize(d, punct) for d in docs]
    X,vocab = vectorize(tokens_list, functions,min_freq)
    clf = LogisticRegression()
    clf.fit(X,labels)
    #clf.predict(X)
    return (clf,vocab)
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    #get the learned coefficients
    #print(len(clf.coef_))
    coef = clf.coef_[0]
    if label==1:
      #coef = clf.coef_[0]
      coef_ind = np.argsort(coef)[::-1][:n]
    else:
      #coef = clf.coef_[0]
      coef_ind = np.argsort(coef)[::1][:n]
    #sort them in descending order
    #coef_ind = np.argsort(coef)[::-1][:n]
    #get names of those features
    #print(type(vocab))
    #print(vocab)
    newvoc = {}
    for k,v in vocab.items():
      newvoc[v] = k
    coef_name = []
    result = []
    for ind in coef_ind:
      coef_name.append(newvoc[ind])
      result.append((newvoc[ind], abs(coef[ind])))
    #coef_name = vocab[coef_ind]
    #result = []

    #result = [(coef_name, coef[coef_ind])]
    return result
    pass
"""

check size of X for pass or not pass the vocab to vec...

"""
def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    test_docs, test_labels = read_data(os.path.join('data','test'))
    punct = best_result['punct']
    functions = best_result['features']
    min_freq = best_result['min_freq']
    #print(min_freq)
    tokens_list = [tokenize(d,punct) for d in test_docs]
    #X_test, vocab = vectorize(tokens_list, functions, min_freq)
    #vectorize the test_data
    # do not need to cal the vocab.
    result = []
    for items in tokens_list:
      result.append(featurize(items, functions))
    data = []
    row = []
    col = []
    for r in range(0,len(result)):
      dic = {}
      for k,v in result[r]:
        dic[k] = v
      for key in vocab.keys():
        if key in dic:
          data.append(dic[key])
          row.append(r)
          col.append(vocab[key])
    datas = np.array(data,dtype='int64')
    rows = np.array(row,dtype='int64')
    cols = np.array(col,dtype='int64')
    l = len(list(vocab.keys()))
    X_test = csr_matrix((datas,(rows, cols)),shape = (len(result), l))
    return test_docs, test_labels,X_test
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    #wronglist store the index of misclassified document.
    wronglist = []
    predictions = clf.predict(X_test)
    for i in range(0, len(predictions)):
      if predictions[i] != test_labels[i]:
        wronglist.append(i)
    prob = clf.predict_proba(X_test)
    #print(type(prob))
    #print(prob.shape)
    prob_list = []
    for element in wronglist:
      if test_labels[element] == 1:
        prob_list.append((element, prob[element][0]))
      else:
        prob_list.append((element, prob[element][1]))
    #print(type(prob_list))
    #print(type(prob_list[0]))
    #print(type(prob_list[0][1]))
    prob_list = sorted(prob_list, key = lambda x: x[1],reverse = True)
    count = n
    while count >0:
      print('\ntruth = %d predicted = %d  proba = %f' %(test_labels[prob_list[n-count][0]], predictions[prob_list[n-count][0]], prob_list[n-count][1]))
      print(test_docs[prob_list[n-count][0]])

      count -= 1
    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
