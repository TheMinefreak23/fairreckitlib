README_Sanaz

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
