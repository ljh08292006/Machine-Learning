import matplotlib.pyplot as plt
import openpyxl as openpyxl
import pandas as pd


raw_data = pd.read_excel("E:/py——file/demo.xlsx", usecols=[1,2,3,4,5,6,7])   # ⑧
raw_data.head()
province_data = raw_data.groupby(['PROVINCE'],as_index=False).sum()
plt.rc("font",family="SimHei",size="14")    # ②设置图形的字体为SimHei，大小为14，目的是避免中文显示乱码的问题
plt.rcParams['axes.unicode_minus'] =False    # ③设置当图形中出现负号时可正常显示
province_bar = province_data.sort_values(['AMOUNT'],ascending=False)

province_bar.plot(kind='bar', x='PROVINCE',y=['AMOUNT','VISITS'], figsize=(10, 4),title='各省份商品销售对比',fontsize=12)
province_barh = province_data.sort_values(['AMOUNT'],ascending=True)

province_barh.plot(kind='barh', x='PROVINCE',y=['AMOUNT','VISITS'], figsize=(10, 4), logx=True,title='各省份商品销售对比',fontsize=10)
raw_data = pd.read_excel("E:\py——file\demo.xlsx")
datetime_data = raw_data.groupby(['DATETIME'],as_index=False).sum()  # ①对raw_data按DATETIME列做分类汇总，汇总计算指标是全部，汇总计算方式是sum求和，得到datetime_data
datetime_data.plot(kind='line', x='DATETIME',y=['AMOUNT','VISITS'], figsize=(10, 4),title='按日销售走势')  # ②调用datetime_data的plot方法展示折线图，整个参数配合与柱形图相同

cate_data = raw_data.groupby(['CATE'],as_index=False)['VISITS'].sum()  # ①将raw_data按CATE列做分类汇总，汇总指标为VISITS，汇总计算方式为求和，得到cate_data
cate_data = cate_data.sort_values(['VISITS'],ascending=False)  # ②cate_data按汇总后的VISITS列倒序排序，目的也是便于按照逻辑顺序展示分布结果
labels = cate_data['CATE']  # ③获得labels数据
cate_data.plot(kind='pie', y='VISITS', figsize=(6, 6),title='VISIT在各个CATE中的分布', labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6,labels=labels,legend=False)  # ④调用cate_data.plot展示饼图
print(datetime_data)

raw_data.plot(kind='scatter',x='AMOUNT', y='VISITS', figsize=(10, 4),title='AMOUNT和MONEY关系')
