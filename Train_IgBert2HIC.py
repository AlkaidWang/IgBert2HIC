import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import random
import math

import time
import datetime 
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience  # 容忍多少个epoch损失不减少
        self.delta = delta  # 最小的损失变化量
        self.best_loss = np.inf  # 记录最好的验证损失
        self.counter = 0  # 计数器，用于记录验证损失没有改善的 Epoch 数
        self.early_stop = False  # 是否提前停止的标志

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss  # 更新最好的损失
            self.counter = 0  # 如果损失改善，重置计数器
        else:
            self.counter += 1  # 如果损失没有改善，计数器加一
            if self.counter >= self.patience:
                self.early_stop = True  # 达到容忍次数，停止训练


#########################################################################
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

##########################################################################

tokeniser = BertTokenizer.from_pretrained("./", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained('./', num_labels=1)

epochs=40
batch_size=16 # 32时测试数据会内存不够
loss_fct = nn.MSELoss()
early_stopping = EarlyStopping(patience=3, delta=0.01)
optimizer = AdamW(model.parameters(), lr=1e-5)         #optimal 5e-5
scheduler = StepLR(optimizer, step_size=10, gamma=0.2)  # 每10个epoch，学习率下降为原来的0.2倍

train_set = pd.read_csv("HIC-1.txt",sep="\t").dropna().drop_duplicates()
train_set["SEQ"]=train_set["VH"].apply(lambda x: ' '.join(x)) + ' [SEP] '+train_set["VL"].apply(lambda x: ' '.join(x))
train_set, valid_set = train_test_split(train_set,test_size=0.1)
print(f'{len(train_set)} training set records, {len(valid_set)} validation set records')

############################################################################

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for i in range(0, len(train_set), batch_size):
    
        # Progress update every 40 batches.
        if i % batch_size == 0 and not i == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, len(train_set), elapsed))
            
            
        # 获取当前批次的文本和标签
        batch_texts = train_set["SEQ"][i:i+batch_size]
        batch_labels = train_set["HIC"][i:i+batch_size]
        batch_labels = batch_labels.tolist() 

        # 将文本转换为BERT输入格式
        encoding = tokeniser.batch_encode_plus(batch_texts, add_special_tokens=True, padding=True, return_tensors="pt",return_special_tokens_mask=True)
        
        # 获取input_ids和attention_mask
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        optimizer.zero_grad()

       # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask,labels=batch_labels)

        # 计算损失
        loss = loss_fct(outputs.logits.squeeze(-1),batch_labels)  # 使用回归损失
        total_train_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()
        
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss/ math.ceil(len(train_set)/batch_size)      
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_valid_loss = 0
    all_labels=[]
    all_preidctions=[]

    # Evaluate data for one epoch
    for i in range(0, len(valid_set), batch_size):
    
        # 获取当前批次的文本和标签
        batch_texts = valid_set["SEQ"][i:i+batch_size]
        batch_labels = valid_set["HIC"][i:i+batch_size]
        batch_labels = batch_labels.tolist() 

        # 将文本转换为BERT输入格式
        encoding = tokeniser.batch_encode_plus(batch_texts, add_special_tokens=True, padding=True, return_tensors="pt",return_special_tokens_mask=True)
        
        # 获取input_ids和attention_mask
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask,labels=batch_labels)
        
        predicted_score=outputs.logits
        loss=loss_fct(predicted_score.squeeze(-1),batch_labels)

        # Accumulate the validation loss.
        total_valid_loss += loss.item()    

        all_labels.extend(batch_labels.cpu().numpy())
        all_preidctions.extend(predicted_score.cpu().numpy())

        
    # Report the final accuracy for this validation run.
    mse = mean_squared_error(all_labels, all_preidctions)
    #print(f"Mean Squared Error: {mse}")

    # 计算 RMSE
    rmse = np.sqrt(mse)
    #print(f"Root Mean Squared Error: {rmse}")

	# 计算 MAE
    mae = mean_absolute_error(all_labels, all_preidctions)
    #print(f"Mean Absolute Error: {mae}")

	# 计算 R2
    r2 = r2_score(all_labels, all_preidctions)
    #print(f"R2 Score: {r2}")

    print("  MSE: {0:.2f}".format(mse))
    print("  RMSE: {0:.2f}".format(rmse))
    print("  MAE: {0:.2f}".format(mae))
    print("  R2: {0:.2f}".format(r2))

    # Calculate the average loss over all of the batches.
    avg_valid_loss = total_valid_loss / math.ceil(len(valid_set)/batch_size) 
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_valid_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_valid_loss,
            'Valid. MSE.': mse,
            'Valid. RMSE.': rmse,
            'Valid. MAE.': mae,
            'Valid. R2.': r2,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    scheduler.step()
    early_stopping(avg_valid_loss)
    if early_stopping.early_stop:
        print("")
        print(f"Early stopping at epoch {epoch_i}")
        break

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

model.save_pretrained('HIC-1')
import pickle
with open('tmp.pkl', 'wb') as file:
    pickle.dump(training_stats, file)

#with open('tmp.pkl', 'rb') as file:
#    training_stats = pickle.load(file)

#######################################################################################

# Display floats with two decimal places.
pd.set_option('display.precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
print(df_stats)
