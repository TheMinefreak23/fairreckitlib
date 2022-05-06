README explains the code in dataloaders.py , config.ini and utility.py and gives examples how to test the functionalities

#load and flitering
>>> import fairreckitlib.dataloaders.dataloaders as dl
>>> loader = dl.get_dataloader("LFM-360K")
>>> loader.load_data()
>>> loader.filter_df({'gender':'m','age':(10,40)})


#testing formating
>>> loader.df_formatter(columns_add=['gender'], columns_remove=['rating'])
#for example if you wanted to check whether remove works correctly, first check what are the attributes then do the drop
>>> loader.ui_data_frame.info()
>>> loader.ui_data_frame.drop(columns=['item'], inplace=True)

#also it is easier to work with your sample dataset. I already made the function in utility. note that 
the function takes the path of the text file and the number of lines to be copied to the sample file.
you could make it as follows:
>>> import fairreckitlib.dataloaders.utility
>>> create_sample("file path", 500)
example of a file path: ../Datasets/LFM_360K/usersha1-artmbid-artname-plays.tsv

#dataloader is flexible enough to read config.ini from wherever(config.ini could be wherever you want).
We can give the path of the config file to the loader (as an arg)
therefore you don't have to use the local config file. For example, if we place config.ini inside config_files folder, we can
instantiate a loader object by loading configs from that file as follows: 
>>> import fairreckitlib.dataloaders.dataloaders as dl
>>> loader = dl.get_dataloader("LFM-360K", "./config_files/")    #if you want to give a local file path leave config_files out
>>> loader.load_data()
* it is also notable that config.ini can be defined out of this repository and be stored anywhere in the server. This fashion, 
by updating the config.ini, we don't need to generate the software and deploy it.

# user-item-rating matrix for 2B
>>> import fairreckitlib.dataloaders.dataloaders as dl
>>> loader = dl.get_dataloader("LFM-2B")
>>> loader.load_data()
>>> df = loader.get_user_item_matrix()

