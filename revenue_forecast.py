import re
import json
import bayes
import random
import numpy as np
import pandas as pd
import ridgeRegression
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics                    #测试时用
from sklearn.model_selection import train_test_split        #测试时用
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")


class Preprocess:
    def __init__(self,train,test):
        self.dataTrain =pd.read_csv(train)
        self.dataTest =pd.read_csv(test)
        #self.data =self.data.sort_values('revenue',ascending= False)       #按利润排序
        self.val =['popularity','runtime','revenue']
        self.data =pd.merge(self.dataTrain,self.dataTest,how='outer')       #合并测试训练集


    def jsonChange_old(self,strs,key):                                      #该方法不能处理cast和crew列
        '''
        函数说明：处理json文本，使其可加载（loads)出来
        :param strs: json文本
        :param key: 要提取信息的字典中的key
        :return: list或者NaT--存放目标信息的list
        '''
        p = ("{'",  "', '",  "': ",  ": '",  "'}",  ", '")
        r = ('{"',  '", "',  '": ',  ': "',  '"}',  ', "')
        if type(strs)==float:
            return []
        strs = re.sub('"', "'", strs)
        for i in range(len(p)):
            strs = re.sub(p[i], r[i], strs)
        strs = re.sub(r'\\', '', strs)
        dict_ = json.loads(strs)
        #print(list(i[key] for i in dict_))
        if dict_:
            return list(i[key] for i in dict_)
        else:
            return []


    def jsonChange(self,strs,key):                      #改进后的处理json格式的方法,采用eval函数
        '''
        函数说明：处理json文本，使其可加载（loads)出来
        :param strs: json文本
        :param key: 要提取信息的字典中的key
        :return: list或者NaT--存放目标信息的list
        '''
        if type(strs)==float:
            return []
        strs = eval(strs)
        if strs:
            return list(i[key] for i in strs)
        else:
            return []


    def dealAbnormal(self,col,method):
        '''
        函数说明：处理异常值，去掉或者替换异常值
        :param col: 列标签名
        :param method: box和3sigma(箱型去异常值和3sigma原则去异常值),两种方法
        :return: self.data--处理后的数据集
        '''
        col0 = self.data[col]
        des = col0.describe()         # 统计数据
        std = des['std']
        mean =des['mean']
        # 3σ原则去异常值
        if method=='3sigma':

            xigema3 = 3 * std                                   #3σ

            #self.data = self.data[abs(col0 - mean) <= xigema3] #直接去掉异常值

            col0[abs(col0 - mean) > xigema3] =col0.mean()       #用去掉异常值后的平均数补上

            num = sum(abs(col0 - mean) > xigema3)

        elif method=='box':

            iqr = des['75%'] - des['25%']                        # 四分位间距 =上四分位数-下四分位数  # 箱型图模型
            upBoundry = des['75%'] + 1.5 * iqr                   # 上界
            lowBoundry = des['25%'] - 1.5 * iqr                  # 下界

            col0[col0 >= upBoundry]= col0.mean()
            col0[col0 <= lowBoundry]= col0.mean()                #popularity没有小于下界的值

            num = sum(col0 < lowBoundry) + sum(col0 > upBoundry) # 异常值数目
            #print("异常值数目：", num)
            #print("异常值占比：", num / col0.shape[0])
        else:print('方法参数不存在！')
        return self.data


    def addMissing(self,col):
        '''
        函数说明：补充缺失值
        :param col: 列元素，和apply函数配合
        :return: 无，直接在原数据集修改
        '''
        col0 = self.data[col]
        if col:
            num =col0.isnull().sum()
        else:
            pass
        #print("<",col,">",r"的缺失值比例:   ",num/self.data.shape[0])
        if num:
            self.data.loc[col0.isnull(),col] = col0.mean()              #用平均值来补缺失值
        #else:
        #    print('没有空值\n')


    def normStand(self,data, method):
        '''
        函数说明：归一化和标准化处理
        :param data: 处理的数据集
        :param method: normal和standard（归一化和标准化），两种方法
        :return: 无，直接在原数据集上修改
        '''
        if method == 'normal':                                          # 进行归一化
            for col in data:                                            # bool型归一、没变化
                max = self.data[col].max(axis=0)
                min = self.data[col].min(axis=0)
                self.data[col] = self.data[col].apply(lambda col: (col - min) / (max - min))

        elif method == 'standard':                                      # 标准化
            for col in data:
                mean = self.data[col].mean()
                std = self.data[col].describe()['std']

                self.data[col] = self.data[col].apply(lambda col: (col - mean) / std)
        else:
            print('方法参数不存在！')


    def showplot(self,data):
        '''
        函数说明：化出数据集后的箱型图
        :return: 无
        '''
        data.boxplot()
        plt.ylabel("ylabel")
        #plt.xlabel("different datasets")
        plt.show()


    def to_dataList(self,data,onehot):
        '''
        函数说明：将pd.Dataframe格式按行封装成数组，用作回归
        :param data: 原数据集
        :param onehot: 集成数据集
        :return: 特征变量x和变量y:revenue
        '''
        x = []; y = []                           # 存放处理后的XY值

        for index,row in onehot.iterrows():
            x.append(list(row))

        for i in data.revenue:
            y.append([i])

        return x,y

        
    def featureEng(self):                        #特征工程
        '''
        函数说明：提取特征变量，转化为二进制列，类似独热编码
        :return: 
        '''
        k = 0
        self.onehot= pd.DataFrame()             #建立特征框架
        
        '''流派提取'''
        genres=[]
        for i in preprocess.data.new_genres:
            for j in i:
                genres.append(j)
        genres=list(pd.DataFrame({'A':genres}).A.value_counts().index)[:19]#19
        #最后一个流派值出现一次，舍去，倒数第二个出现84次
        #genres = [18, 35, 53, 28, 10749, 80, 12, 27, 878, 10751, 14, 9648, 16, 36, 10752, 10402, 99, 37, 10769, 10770]
        for i in genres:
            self.onehot['genre_'+str(i)]= self.data.new_genres.apply(lambda col:1 if i in col else 0)


        '''出产国家'''
        countries = []
        for i in self.data.countries:
            if type(i) != float:
                for j in i:
                    #if j not in countries:
                    countries.append(j)
        countries = list(pd.DataFrame({'A': countries}).A.value_counts().index)[:46]#第46名出现过5次
        #列中最后一个出现10次，有80个左右国家，后面的都是只出现一两几次，舍去
        #countries= ['US', 'IN', 'KR', 'CA', 'GB', 'AT', 'DE', 'FR', 'NZ', 'JP', 'IE', 'IT', 'IL', 'BE', 'CN', 'BR', 'AR', 'CL', 'PE', 'CZ', 'HK', 'RU', 'ES', 'TR', 'AU', 'SE', 'LU', 'ZA', 'CH', 'IR', 'MA', 'NL', 'PH', 'IS', 'DK', 'TW', 'MN', 'HU', 'RS', 'MX', 'RO', 'GR', 'AE', 'PR', 'FI', 'KH', 'NO', 'PL', 'NA', 'BA', 'CS', 'PK', 'DZ', 'ET', 'QA', 'TN', 'PT', 'BG', 'SI', 'UA', 'SA', 'JO', 'HR', 'SG', 'UY', 'PY', 'MR', 'MT', 'CR', 'BS', 'BF', 'GH', 'ID']
        for i in countries:
            self.onehot['countries_'+i]= self.data.countries.apply(lambda col:1 if i in col else 0)


        '''语言'''
        languages = []
        for i in self.data.languages:
            for j in i:
                #if j not in languages:
                languages.append(j)
        languages = list(pd.DataFrame({'A': languages}).A.value_counts().index) [:40]#第40名出现8次
        #有98中语言，列中最后一个出现9次，后面的舍去
        #languages= ['en', 'hi', 'ko', 'ar', 'ru', 'sv', 'de', 'fr', 'it', 'ja', 'he', 'pt', 'la', 'zh', 'es', 'nl', 'cn', 'qu', 'cs', 'ta', 'te', 'pl', 'tr', 'hu', 'el', 'ga', 'fa', 'th', 'ca', 'tl', 'da', 'bn', 'info', 'sh', 'gd', 'yi', 'af', 'hy', 'pa', 'bg', 'sw', 'no', 'mr', 'bo', 'fi', 'ur', 'sq', 'ro', 'ln', 'my', 'id', 'vi', 'am', 'sk', 'xx', 'uk', 'eo', 'eu', 'sa', 'ny', 'st', 'xh', 'zu', 'mi', 'ml', 'so', 'lo', 'sr', 'is', 'wo', 'et', 'ps', 'si', 'hr', 'kw', 'nv', 'gu', 'ku', 'km']
        for i in languages:
            self.onehot['languages_'+i]= self.data.languages.apply(lambda col:1 if i in col else 0)


        '''原始语言'''
        orlanguages= []
        for i in self.data.original_language:
            #if i not in orlanguages:
            orlanguages.append(i)
        orlanguages = list(pd.DataFrame({'A': orlanguages}).A.value_counts().index)
        #第20名出现5次共有43左右种语言
        for i in orlanguages:
            self.onehot['orlanguages_'+i]= self.data.original_language.apply(lambda col:1 if i in col else 0)


        '''出产公司'''
        companies = []
        for i in self.data.companies:
            for j in i:
                #if j not in companies:
                companies.append(j)
        companies = list(pd.DataFrame({'A': companies}).A.value_counts().index) [:80]
        #第250名出现10次
        #有2307间公司出现不止1次，1347不止两次，920不止3次，691间不止4次，549不止5次，449不止6次，347不止7次，310不止8次
        for i in companies:
            self.onehot['companies_'+str(i)]=self.data.companies.apply(lambda col:1 if i in col else 0)


        '''关键词'''
        words = []
        for i in self.data.keywords:
            # if type(i)!=float:
            for j in i:
                words.append(j)
        keywords = list(pd.DataFrame({'a': words}).a.value_counts().index)[:380]
        #第150名出现过40次
        for i in keywords :
            self.onehot['keywords_'+str(i)]=self.data.keywords.apply(lambda col:1 if i in col else 0)
            
            
        '''演员名字'''
        castname=[]
        for i in self.data.cast_name:
            if type(i)!=float:
                for j in i:
                    castname.append(j)
        castname = list(pd.DataFrame({'a': castname}).a.value_counts().index)[:80]
        #第80名的出现过35次
        for i in castname:
            self.onehot['cast_name_' + str(i)] = self.data.cast_name.apply(lambda col: 1 if i in col else 0)


        '''产品团队成员'''
        crew =[]
        for i in self.data.crew:
            if type(i)!=float:
                for j in i:
                    crew.append(j)
        crew = list(pd.DataFrame({'a': crew}).a.value_counts().index)[:180]
        #第180名的出现过30次
        for i in crew:
            self.onehot['crew_' + str(i)] = self.data.crew.apply(lambda col: 1 if i in col else 0)



    def __call__(self):
        ''''''
        
        '''异常缺失数据的处理，平滑处理'''
        self.data.release_date[3828:3829]='2001/3/20'                          #这一列缺失值很多，手动补上
        
        self.data['date_month'] = pd.to_datetime(self.data['release_date'], errors='coerce', dayfirst=True).dt.month.apply(np.log)
        self.data['date_year'] = pd.to_datetime(self.data['release_date'], errors='coerce', dayfirst=True).dt.year.apply(np.log)
        self.data['date_year'] =self.data.date_year.apply(lambda x:x-100 if x%100>=19 else x)
        self.data['revenue'] = self.data['revenue'].apply(np.log)              #revenue由于太大，而且有着价格的性质，应做平滑处理
        
        for i in self.val:                                                     #添加缺失值、替补异常值
            self.addMissing(i)
            self.dealAbnormal(i,method='3sigma')                               #3sigma比box好一点
            
        self.data['log_runtime'] = self.data['runtime'].apply(np.log)          #取对数处理时间数据
        self.normStand(['popularity','log_runtime'],'normal')                  #使用归一化处理数据
        
        '''json格式的处理，提取出个特征对应的ID'''
        self.data['new_genres'] =self.data.genres.apply(self.jsonChange,key='id')
        self.data['countries'] = self.data.production_countries.apply(self.jsonChange, key='iso_3166_1')
        self.data['companies'] = self.data.production_companies.apply(self.jsonChange, key='id')
        self.data['languages'] = self.data.spoken_languages.apply(self.jsonChange, key='iso_639_1')
        self.data['cast_name'] = self.data.cast.apply(self.jsonChange,key='id')
        self.data['keywords'] =self.data.Keywords.apply(self.jsonChange,key='id')
        self.data['crew'] = self.data.crew.apply(self.jsonChange,key='id')

        self.data.languages[3828:3829]=['en']                   #这一列缺失值很多，手动补上
        self.data.countries[3828:3829]=['US']                   #这一列缺失值很多，手动补上

        #self.data.cast =self.data.cast.apply(self.jsonChange,key='cast_id')
        #self.data.crew =self.data.crew.apply(self.jsonChange,key='credit_id')
        #print(self.data.crew)

        '''编写二进制列'''
        self.featureEng()

        '''将data里面的一些特征加入onehot里面，方便一起训练'''
        xy = ['log_runtime', 'popularity', 'date_year', 'date_month', 'revenue']
        for i in xy[:-1]:
            self.onehot[i]= self.data[i]

        '''分割测试、训练数据集'''
        #self.data=self.data[:3000]
        #self.onehot =self.onehot[:3000]
        self.dataTest= self.data[3000:]
        self.onehotTest= self.onehot[3000:]

class mySklearn:

    def __init__(self,Preprocess):

        self.Preprocess =Preprocess
        self.data =Preprocess.data

    def ridge(self):
        '''预测'''
        x, y = preprocess.to_dataList(preprocess.data[:3000], preprocess.onehot[:3000])

        test_size = 0.25

        data_class_list = list(zip(x, y))          #zip压缩合并，将数据与标签对应压缩

        random.shuffle(data_class_list)                             #将data_class_list乱序

        index = int(len(data_class_list) * test_size) + 1           #训练集和测试集切分的索引值

        train_list = data_class_list[index:]                        #训练集

        test_list = data_class_list[:index]                         #测试集

        train_data_list, train_class_list = zip(*train_list)        #训练集解压缩

        test_data_list, test_class_list = zip(*test_list)           #测试集解压缩


        r2 ,yPred , coef= ridgeRegression.ols(train_data_list, train_class_list)

        yPred = []

        for i in range(len(test_data_list)):
            yPred.append(np.dot(coef, test_data_list[i]))  # 将的到得系数矩阵和x相乘再相加（点积），得到y的预测值

        print('交叉验证R2:', metrics.r2_score(test_class_list, yPred))
        print('交叉验证RMSE:', np.sqrt(metrics.mean_squared_error(test_class_list, yPred)))

        return metrics.r2_score(test_class_list, yPred) , np.sqrt(metrics.mean_squared_error(test_class_list, yPred)), coef


    def dmTree(self):

        self.data = self.data.drop(axis=1, index=[5398])
        self.data = self.data[:3000]

        self.data.title[self.data.title.isnull()] = 'a'
        self.data.tagline[self.data.tagline.isnull()] = 'a'
        self.data.overview[self.data.overview.isnull()] = 'a'

        self.text_data = self.data[['title','tagline','overview']]#文本类型
        self.class_text = self.data['new_genres']
        self.text = [['title'],['tagline'],['overview']]

        word_list =[]
        sentence_list = []
        status = 0
        preList = []
        classList = []
        #for i in self.text:
        for kind in self.text_data:
            if status:
                k =0
                for sentence in self.text_data[kind]:
                    sentence_list[k].extend(bayes.textParse(sentence))#textParse将句子作词汇分割，然后横向加入列表里面
                    k=k+1

            else:
                status = 1#刚开始进入这里，
                kk =0
                for sentence in self.text_data[kind]:
                    kk=kk+1
                    sentence_list.append(bayes.textParse(sentence))#对空列表进行赋值，下一步就进入if语句，进行横向添加，将三种文本类型合并起来

        #vocabList = bayes.createVocabList(sentence_list)  # 创建词汇表，不重复

        mlb = MultiLabelBinarizer()#多标签贝叶斯

        self.yTrain = mlb.fit_transform(self.class_text)#label的转换，转换成稀疏矩阵  01矩阵


        data_class_list = list(zip(sentence_list, self.yTrain))  # zip压缩合并，将数据与标签对应压缩

        random.shuffle(data_class_list)  # 将data_class_list乱序

        index = int(len(data_class_list) * 0.25) + 1  # 训练集和测试集切分的索引值

        train_list = data_class_list[index:]  # 训练集

        test_list = data_class_list[:index]  # 测试集

        train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩

        test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩


        all_words_dict = {}  # 统计训练集词频

        for word_list in train_data_list:#在对训练集的每一个单词列表

            for word in word_list:#对单词列表的每一个单词

                if word in all_words_dict.keys():

                    all_words_dict[word] += 1#对单词进行计数

                else:

                    all_words_dict[word] = 1

        # 根据键的值倒序排序

        all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)#词汇量排序

        all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩

        all_words_list = list(all_words_list)  # 转换成列表

        # 生成stopwords_set

        stopwords_file = './stopwords.txt'

        stopwords_set = bayes.MakeWordsSet(stopwords_file)#停词表


        test_accuracy_list = []

        deleteNs = range(8000, 12000, 100)  # 0 20 40 60 ... 980

        #for deleteN in deleteNs:
        feature_words = bayes.words_dict(all_words_list,11200, 20, stopwords_set)

        train_feature_list, test_feature_list = bayes.TextFeatures(train_data_list, test_data_list, feature_words)#制造单词特征

        test_accuracy, pre , single_accuracy = bayes.TextClassifier(train_feature_list, test_feature_list, np.array(train_class_list), np.array(test_class_list))#进行分类

        preList.append(np.array(pre))
        classList.append(np.array(test_class_list))

        test_accuracy_list.append(test_accuracy)
        print('test_accuracy_list: ',test_accuracy_list)
        print('single_accuracy: ',single_accuracy)
        '''
        投票
        h = preList[0] + preList[1] + preList[2] + np.array(test_class_list)
        num =0
        finalPre = np.zeros(preList[0].shape)
        for i in h:
            print(i)
        for i in h:
            for j in i:
                if 4 <= j:
                    num = num + 1
                    break
        print(num/len(h))
        '''
        '''

        plt.figure() 

        plt.plot(deleteNs, test_accuracy_list) 

        plt.title('Relationship of deleteNs and test_accuracy')

        plt.xlabel('deleteNs')

        plt.ylabel('test_accuracy')

        plt.show() '''


    '''k近邻法，作分类'''
    def knn(self,inX, dataSet, labels, k ,multi):
        '''
        函数说明：电影流派分类时使用
        :param inX: 要进行分类的向量
        :param dataSet: 训练集
        :param labels: 训练集对应的标签
        :param k: 近邻数目
        :param multi: 是否多标签
        :return: 
        '''
        dataSetSize = dataSet.shape[0]
        #print('dataSetSize',dataSetSize)

        diffMat = np.tile(inX, (dataSetSize,1)) - dataSet           #竖向瓦矩阵，与数据集进行相减，求两点距离的第一步
        #print('diffMat',diffMat)

        sqDiffMat = diffMat**2  #（x1-x2）^2
        #print('sqDiffMat',sqDiffMat)

        sqDistances = sqDiffMat.sum(axis=1) #x,y的平方和相加，横向相加
        #print('sqDistances',sqDistances)

        distances = sqDistances**0.5    #开根号
        #print('distances',distances)

        sortedDistIndicies =distances.argsort()                     #排大小，输出排序后每个数的索引
        #print('sortedDistIndicies: ',sortedDistIndicies)

        if multi=='True':
            class_ =[]
            for i in range(k):
                voteIlabel = labels[sortedDistIndicies[i]]           #从标签里面拿出该值对应的标签
                #print(voteIlabel)
                for j in voteIlabel:
                    class_.append(j)
            class_freq =pd.DataFrame({'A':class_}).A.value_counts(1)  #series类型
            #print(class_freq)
            the_most_likely =list(class_freq.index)[0]
            return the_most_likely

        elif multi=='False':
            classCount = {}  # 存放标签及其数目
            for i in range(k):
                voteIlabel = labels[sortedDistIndicies[i]]  # 从标签里面拿出该值对应的标签
                #print(voteIlabel)
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
                #print('classCount', classCount)

            sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
            #print('sortedClassCount',sortedClassCount)
            return sortedClassCount[0][0]

        else:print('方法参数错误！')

    def train(self):

        x_train,x_test,y_train,y_test =train_test_split(self.data,self.data['revenue'],random_state=1)

        linreg =linear_model.LinearRegression()#普通回归
        model = linreg.fit(x_train, y_train.astype('int'))
        y_pred = linreg.predict(x_test)
        print("MSE:", metrics.mean_squared_error(y_test, y_pred.astype('int')))  # 均方差
        print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred.astype('int'))))
        print('Variance score: %.2f' % metrics.r2_score(y_test, y_pred.astype('int')))  # r值
        print('R-squared: {:.2f}'.format(model.score(x_test, y_test)))  # r值，线性相关度，准确率


if __name__=='__main__':
    info=[]                                     #存放训练分数，预测集最大值等。。。,调试时用
    #for company_num in range(55,68,2):         #变化加入拟合的公司数目作拟合，看最优加入公司数目是多少,调试时用
    #for k in range(22):                        #调试时用
    submit ='sample_submission.csv'
    test ='test.csv'
    train ='train.csv'
    r2list=[]
    RMSElist = []#用于画图

    '''实例化并预处理'''
    preprocess =Preprocess(train,test)
    preprocess()                                #call魔法方法

    #岭回归
    #for hh in range(320, 420, 5):  # 调试用，调试加入特征的数量、岭回归的lamada值等
    forecast = mySklearn(preprocess)
    '''岭回归
    r2, RMSE ,coef = forecast.ridge()
    r2list.append(r2)
    RMSElist.append(RMSE)
    '''

    '''贝叶斯'''
    forecast.dmTree()

    plt.plot(r2list)
    # plt.plot(RMSElist)
    # plt.savefig('./company/(60,86,2)2.png')
    #plt.show()

    '''
    #用预测出来的函数计算测试集
    xTest=[];yPred= []                          #存放将要计算出来的预测值
    for i,row in preprocess.onehotTest.iterrows():
        xTest.append(list(row))                 #将测试值的每一排取出来，放进XTest里面，一排一排放，变成二维列表
    for i in range(len(xTest)):
        yPred.append(np.dot(coef, xTest[i]))  # 将的到得系数矩阵和x相乘再相加（点积），得到y的预测值

    #log_revenue取指数，计算revenue
    yPred =map(np.exp,yPred)                    #还原预测值，本来已经去对数log
    yPred =[i for i in yPred]
    test_revenue =pd.read_csv(submit)
    test_revenue.revenue =pd.DataFrame({'revenue':yPred}).revenue   #准备存放数据

    #输出到CSV
    #test_revenue.to_csv('submit_revenue.csv')
    '''



    '''流派的分类，不完整。。。
    preprocess.normStand(['revenue'],method='normal')
    dataSet =np.array(preprocess.data[:3000][['popularity','log_runtime','revenue']])
    label =list(preprocess.data.new_genres)
    x,y=preprocess.to_dataList(preprocess.data,preprocess.data[3000:][['popularity','log_runtime','revenue']])
    j=0
    num=0
    for i in x:
        correct =b.knn(i,dataSet,label,k=30,multi='True')
        if correct in preprocess.data[3000:].new_genres.values[j]:
            j+=1
            num+=1
    print(num)
    print('正确率：',num/len(preprocess.data[3000:]))
    '''