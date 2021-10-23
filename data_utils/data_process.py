import csv
import os
import pandas as pd
import glob
import shutil
from random import randint, randrange

'''
Get the all images files from the folder
'''
def get_images_list(dirName,endings=['.jpg','.jpeg','.png','.JPG']):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for i,ending in enumerate(endings):
        if ending[0]!='.':
            endings[i] = '.'+ending
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_images_list(fullPath,endings)
        else:
            for ending in endings:
                if entry.endswith(ending):
                    allFiles.append(fullPath)               
    return allFiles  



class ImagePreProcess():

    def __init__(self,val_split=0.1):
        self.val_split = val_split

    '''
    Give a unique name for the images
    '''
    def create_unique_id(self,df,image_folder):

        index_no = randint(100, 999) 
        data_dict={}
        dir_path = os.path.dirname(image_folder)

        input_paths = get_images_list(image_folder)

        for count, filename in enumerate(input_paths):
            dst = str(index_no) + str(count) + ".jpg"
            data_dict[os.path.basename(filename)] = dst
            dst = os.path.join(image_folder, dst)
            os.rename(filename, dst)

        df["image"] = df["image"].map(data_dict)

        return df


    '''
    Convert CSV data format to Yolo 
    '''

    def convert_Input_CSV_to_yolo(self,vott_df,labeldict=dict(zip(['Object'],[0,])),
                                path='',train_name='train.txt',val_name='val.txt',
                                abs_path=False,target_name = 'data_file_out.txt'):


        if not 'code' in vott_df.columns:
            vott_df['code']=vott_df['label'].apply(lambda x: labeldict[x])
        for col in vott_df[['xmin', 'ymin', 'xmax', 'ymax']]:
            vott_df[col]=(vott_df[col]).apply(lambda x: round(float(x)))

        #Create Yolo Text file
        last_image = ''
        txt_file = ''

        for index,row in vott_df.iterrows():
            if not last_image == row['image']:
                if abs_path:
                    txt_file +='\n'+row['image_path'] + ' '
                else:
                    txt_file +='\n'+os.path.join(path,row['image']) + ' '
                txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax','code']].tolist())])
            else:
                txt_file += ' '
                txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax','code']].tolist())])
            last_image = row['image']
        file = open(target_name,"w") 
        file.write(txt_file[1:]) 
        file.close()
        dataset = pd.read_csv(target_name,delimiter="\n")
        dataset_copy = dataset.copy()
        train_set = dataset_copy.sample(frac=1-self.val_split, random_state=0)
        val_set = dataset_copy.drop(train_set.index)

        train_set.to_csv(train_name, header=True, index=False, sep='\n', mode='w')
        val_set.to_csv(val_name, header=True, index=False, sep='\n', mode='w')

        return True


    '''
    Vott CSV data format pre processing steps
    '''
    def csv_data_process(self,folder_path,
                        class_names='class.names',
                        out_folder='process_data',
                        model_folder='data_models',
                        train_file ='train.txt',val_file = 'val.txt'
                        ):

        data_set = os.path.basename(folder_path)
        model_folder = os.path.join(model_folder,data_set)
        csv_file = None

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        for fname in os.listdir(folder_path):
            if fname.endswith('.csv'):
                csv_file = fname

        output_path = os.path.join(model_folder, out_folder)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        shutil.copytree(folder_path, output_path)

        csv_pd = pd.read_csv(os.path.join(folder_path,csv_file))
        multi_df = self.create_unique_id(csv_pd, output_path)

        labels = multi_df['label'].unique()
        labeldict = dict(zip(labels,range(len(labels))))
        multi_df.drop_duplicates(subset=None, keep='first', inplace=True)

        self.convert_Input_CSV_to_yolo(multi_df,labeldict,path=os.path.abspath(output_path),
                                    train_name=os.path.join(model_folder,train_file),val_name=os.path.join(model_folder,val_file),
                                    target_name=os.path.join(model_folder,'data_file_out.txt'))

        file = open(os.path.join(model_folder,class_names),"w") 
        SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])
        for elem in SortedLabelDict:
            file.write(elem[0]+'\n') 
        file.close()

        return model_folder

