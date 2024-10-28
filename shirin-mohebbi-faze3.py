import numpy as np
import pandas as pd
import sys
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import sklearn.metrics as sklearn
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class RuleBaseSys():
  def __init__(self):
    with open('covid.csv') as f:
      content = [line.strip().split(',') for line in f] 
    features = content[1:]
    for row in features:
      if row[15] == '2':
        row[15] = '1'
      del row[0] #gender

    features = [[float(string) for string in inner] for inner in features]
    features = np.array(features)
    max1 = np.max(features, axis=0)
    min1 = np.min(features, axis=0)
    for i in range(0, len(features)):
      features[i] = ((features[i] - min1) / (max1 - min1)) * np.array([100] * 15)

    features = sorted(features, key=lambda x: x[14], reverse=True)
    self.features = np.array(features[:44] + features[59:80])
    self.test = np.array(features[44:59] + features[80:])

  def feature_fuzzification(self):
    featuresFuzzySets = []
    features = np.copy(self.features)
    minFeatures = np.min(features, axis=0)
    maxFeatures = np.max(features, axis=0)
    for i in range(14):
      x_domain = np.arange(minFeatures[i], maxFeatures[i]+1, 1)
      fiSets = []
      step = (maxFeatures[i]+1 - minFeatures[i]) / 2
      # Generate fuzzy membership functions
      fiSets.append(fuzz.trimf(x_domain, [minFeatures[i], minFeatures[i], minFeatures[i]+step])) #low
      fiSets.append(fuzz.trimf(x_domain, [minFeatures[i], minFeatures[i]+step, minFeatures[i]+2*step])) #medium
      fiSets.append(fuzz.trimf(x_domain, [minFeatures[i]+step, minFeatures[i]+2*step, minFeatures[i]+2*step])) #high
      featuresFuzzySets.append(fiSets)

    plt.plot(x_domain, featuresFuzzySets[0][0], 'g', linewidth=1.5, label='low')
    plt.plot(x_domain, featuresFuzzySets[0][1], 'y', linewidth=1.5, label='med')
    plt.plot(x_domain, featuresFuzzySets[0][2], 'r', linewidth=1.5, label='high')
    t = "features domain fuzzy sets"
    plt.title(t)
    plt.legend()
    plt.show()
    self.featuresFuzzySets = featuresFuzzySets

  def extraxct_rule_from_patients(self):
    rules = []
    covidR = []
    notCovidR = []
    patients = np.copy(self.features)
    minFeatures = np.min(patients, axis=0)
    maxFeatures = np.max(patients, axis=0)
    for p in patients:
      ri = []
      for i in range(len(p) - 1):
        x_domain = np.arange(minFeatures[i], maxFeatures[i]+1, 1)
        lo_member = fuzz.interp_membership(x_domain, self.featuresFuzzySets[i][0], p[i])
        md_member = fuzz.interp_membership(x_domain, self.featuresFuzzySets[i][1], p[i])
        hi_member = fuzz.interp_membership(x_domain, self.featuresFuzzySets[i][2], p[i])
        memList = [lo_member, md_member, hi_member]
        membership = memList.index(max(memList))
        ri.append(membership)
      ri.append(p[len(p) - 1]) #label
      if ri in rules:
        continue
      if p[len(p) - 1] == 100:
        covidR.append(ri)
      else:
        notCovidR.append(ri)
      rules.append(ri)

    self.rules = rules
    self.covidRules = covidR
    self.notCovidRules = notCovidR

  def unionAlphacuts(self, alphaCuts):
    res = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
      for cut in alphaCuts:
        if cut[i] == 1:
          res[i] = 1
          break
    return res

  def defuzzy_alpha_cut(self, alphaCut):
    sum1 = 0
    for i in range(10):
      sum1 += (i+1) * alphaCut[i]
    return sum1/55

  def printRule(self, r):
    textRule = ""
    for i in range(len(r) - 1):
      if r[i] == 0:
        textRule += ('if feature ' + str(i) + " is low")
      elif r[i] == 1:
        textRule += ('if feature ' + str(i) + " is medium")
      else:
        textRule += ('if feature ' + str(i) + " is high")
      if i != len(r) - 2:
        textRule += " and "
    if r[len(r) - 1] == 100:
      textRule += " then patient has covid"
    else:
      textRule += " then patient doesn't have covid"
    print(textRule)

  def plot_covid_domain_fuzzy_sets(self):
    q_domain = np.arange(0, 11, 1)
    qFuzzyHighCovid = fuzz.trimf(q_domain, [5, 10, 10])
    qFuzzyLowCovid = fuzz.trimf(q_domain, [0, 0, 5])
    plt.plot(q_domain, qFuzzyLowCovid, 'g', linewidth=1.5, label='low')
    plt.plot(q_domain, qFuzzyHighCovid, 'r', linewidth=1.5, label='high')
    t = "covid domain fuzzy sets"
    plt.title(t)
    plt.legend()
    plt.show()
    
  def predict(self, p):
    features = np.copy(self.features)
    minFeatures = np.min(features, axis=0)
    maxFeatures = np.max(features, axis=0)
    q_domain = np.arange(1, 11, 1)
    qFuzzyHighCovid = fuzz.trimf(q_domain, [5, 10, 10])
    alphaCutsCovid = []
    for rule in self.covidRules:
      memberships = []
      for i in range(len(rule)-1):
        x_domain = np.arange(minFeatures[i], maxFeatures[i]+1, 1)
        fuzzySetToFire = self.featuresFuzzySets[i][rule[i]]
        m = fuzz.interp_membership(x_domain, fuzzySetToFire, p[i])
        memberships.append(m)
      minMembership = min(memberships)
      alphaCutsCovid.append(fuzz.lambda_cut(qFuzzyHighCovid, minMembership))
    defuz = self.unionAlphacuts(alphaCutsCovid)
    covidHigh = self.defuzzy_alpha_cut(defuz)
    # print("defuz0", defuz, covidHigh)
    q_domain = np.arange(0, 10, 1)
    qFuzzyLowCovid = fuzz.trimf(q_domain, [0, 0, 5])
    alphaCutsNotCovid = []
    for rule in self.notCovidRules:
      memberships = []
      for i in range(len(rule)-1):
        x_domain = np.arange(minFeatures[i], maxFeatures[i]+1, 1)
        fuzzySetToFire = self.featuresFuzzySets[i][rule[i]]
        m = fuzz.interp_membership(x_domain, fuzzySetToFire, p[i])
        memberships.append(m)
      minMembership = min(memberships)
      alphaCutsNotCovid.append(fuzz.lambda_cut(qFuzzyLowCovid, minMembership))
    defuz = self.unionAlphacuts(alphaCutsNotCovid)
    covidLow = self.defuzzy_alpha_cut(defuz)
    # print("defuz1", defuz, covidLow)
    if covidHigh > covidLow:
      return 1
    return 0

  def get_results(self):
    pred = []
    true = []
    print("sample rules")
    self.printRule(self.covidRules[0])
    print("")
    self.printRule(self.notCovidRules[0])
    print("")
    self.plot_covid_domain_fuzzy_sets()
    for p in self.test:
      pred.append(self.predict(p))
      if p[14] == 100:
        true.append(1)
      else:
        true.append(0)
    cm = sklearn.confusion_matrix(true, pred)
    trueNeg = cm[0][0]
    falseNeg = cm[1][0]
    falsePos = cm[0][1]
    truePos = cm[1][1]
    accuracy = ((trueNeg + truePos) / len(self.test)) * 100
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    fMeasure = 2 * ( (precision * recall) / (precision + recall) )
    print ("results: ")
    print ("accuracy: ", accuracy)
    print ("precision: ", precision)
    print ("recall: ", recall)
    print ("fmeasure: ", fMeasure)


r = RuleBaseSys()
r.feature_fuzzification()
r.extraxct_rule_from_patients()
r.get_results()

