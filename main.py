from dataloaders import dataloaders as dl
df = dl.dataloader('lfm_360k')
print(f"{df[list(df.keys())[0]].info()}")

#PS C:\Users\sanaz\OneDrive\Documents\GitHub\fair-rec-kit-lib> python .\main.py
#for testing instead of lfm_360k we can put ml_100k_u . One hint. if we write the name of the file from config.ini compeltely then it takes less memory
