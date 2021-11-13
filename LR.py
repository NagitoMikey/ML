import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc


train_bkf = pd.read_table('C:\\Users\\GWC\\Desktop\\train.txt', header=None)  # 读取文件
test_bkf = pd.read_table('C:\\Users\\GWC\\Desktop\\test.txt', header=None)
train_mmi = pd.read_table('C:\\Users\\GWC\\Desktop\\train_mmi.txt', header=None)
test_mmi = pd.read_table('C:\\Users\\GWC\\Desktop\\test_mmi.txt', header=None)

train_bkf_label = train_bkf[500]  # 获取训练集和测试集的特征和标签
train_bkf.drop([500], axis=1, inplace=True)
test_bkf_label = test_bkf[500]
test_bkf.drop([500], axis=1, inplace=True)
train_mmi_label = train_mmi[34]
train_mmi.drop([34], axis=1, inplace=True)
test_mmi_label = test_mmi[34]
test_mmi.drop([34], axis=1, inplace=True)

lr_model_bkf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)  # 调用模型，但是并未经过任何调参操作，使用默认值
lr_model_bkf.fit(train_bkf, train_bkf_label)# 训练模型
lr_model_mmi = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)
lr_model_mmi.fit(train_mmi, train_mmi_label)
predictions_bkf = lr_model_bkf.predict(test_bkf)
predictions_mmi = lr_model_mmi.predict(test_mmi)

print(lr_model_bkf.score(test_bkf, test_bkf_label))# 获取测试集的评分
print(lr_model_mmi.score(test_mmi, test_mmi_label))
confusion_matrix_bkf = confusion_matrix(predictions_bkf,test_bkf_label)
confusion_matrix_mmi = confusion_matrix(predictions_mmi,test_mmi_label)
print(confusion_matrix_bkf)
print(confusion_matrix_mmi)

predictions_bkf=lr_model_bkf.predict_proba(test_bkf)# 获得测试集上训练模型得到的结果
predictions_mmi=lr_model_mmi.predict_proba(test_mmi)
false_positive_rate_bkf, recall_bkf, thresholds_bkf = roc_curve(test_bkf_label, predictions_bkf[:
, 1])# 获得真假阳性率
false_positive_rate_mmi, recall_mmi, thresholds_mmi = roc_curve(test_mmi_label, predictions_mmi[:
, 1])
roc_auc_bkf=auc(false_positive_rate_bkf,recall_bkf)# 作图
roc_auc_mmi=auc(false_positive_rate_mmi,recall_mmi)
plt.title('ROC')
plt.plot(false_positive_rate_bkf, recall_bkf, 'b', label='BKF_AUC = %0.2f' % roc_auc_bkf)
plt.plot(false_positive_rate_mmi, recall_mmi, 'r-.',label='MMI_AUC = %0.2f' % roc_auc_mmi)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

