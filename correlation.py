import re
import pdb
import pandas as pd
import sys


def acc_length_corr(file_name):
  data = {100:0,500:0,1000:0,2000:0}
  count_100=0
  count_500=0
  count_1000=0
  count_2000=0
  with open(file_name) as f:
    for line in f:
        line = line.rstrip()
        #pdb.set_trace()
        c = line.split()
        len = int(c[1].split('.')[1])
        if len < 100:
          data[100] +=1
          if c[0] == c[1].split('.')[0].split('/')[-1]:
            count_100+=1
        if len >= 100 and len < 500:
          data[500] +=1
          if c[0] == c[1].split('.')[0].split('/')[-1]:
            count_500+=1 
        if len >= 500 and len <1000 :
          data[1000] +=1
          if c[0] == c[1].split('.')[0].split('/')[-1]:
            count_1000+=1
        if len > 1000 :
          data[2000] +=1
          if c[0] == c[1].split('.')[0].split('/')[-1]:
            count_2000+=1     

  data[100] = count_100/data[100]
  data[500] = count_500/data[500]
  data[1000] = count_1000/data[1000]
  data[2000] = count_2000/data[2000]  
  return data 
  
  
def main():  
	gen = sys.argv[1]
	spam = sys.argv[2]
	data_gen = acc_length_corr(gen)
	data_spam = acc_length_corr(spam)

	df_gen = pd.DataFrame(data_gen.items())
	df_spam = pd.DataFrame(data_spam.items())

	import matplotlib.pyplot as plt
	plt.scatter(df_gen[0],df_gen[1],label='gen')
	plt.scatter(df_spam[0],df_spam[1],label='spam')
	plt.xlabel('Sentence length (#words)')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid('--')
	plt.savefig('corelation.png', bbox_inches='tight')

if __name__=="__main()__":
	main()