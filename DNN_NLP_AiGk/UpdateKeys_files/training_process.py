'''
Training Process
'''
from classification_utils import *
from text_rank_utils import *


df =pd.read_pickle('data_pickle.pkl')
df_total = data_processing(df)
df_train, df_test = split_data(df_total)
X_train, X_val, Y_train, Y_val = split_train_data(df_train)
vocab = create_vocab(X_train,True)
vocab =  load_vocab('vocab.txt')

X_train = process_docs(X_train, vocab)
X_val = process_docs(X_val, vocab)
X_train,X_val,tokenizer_object = pipeline_tokenizer_matrix(X_train,X_val)
X_train,X_val,pca_object = pipeline_pca(X_train,X_val,300)
history,acc,model = pipeline_modelling_bow(X_train,X_val,Y_train,Y_val,0)
print('validation_accuracy',acc)
Y_pred_test_b,Y_test,acc_test = testing_data_bow(model,df_test,tokenizer_object,pca_object,vocab)
print('testing_accuracy',acc_test)


with open('model_objects.pkl', 'wb') as f:
    pickle.dump([pca_object,tokenizer_object], f)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

#loadmodel
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

Y_pred_test_b,Y_test,acc_test = testing_data_bow(loaded_model,df_test,tokenizer_object,pca_object,vocab)
print('loaded_model_testing_accuracy',acc_test)

