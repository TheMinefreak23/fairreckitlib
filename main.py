from dataloaders import dataloaders as dl

# Running the following 2 lines will read in the dataset that is passed as argument
# and will print info about the dataframe
df = dl.dataloader('lfm_360k_usersha1-artmbid-artname-plays')
print(f"{df[list(df.keys())[0]][:2]}")

#PS C:\Users\sanaz\OneDrive\Documents\GitHub\fair-rec-kit-lib> python .\main.py
#for testing instead of lfm_360k we can put ml_100k_u . One hint. if we write the name of the file from config.ini compeltely then it takes less memory
