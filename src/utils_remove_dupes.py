
import pandas as pd


df = pd.read_csv("../" + "symbols_df_bigwill_usdt.csv")

lst = ['XRP', 'BCH', 'DOGE', 'AAVE', 'ATOM', 'SUSHI', 'SHIB', 'EGLD', 'AR', 'PEOPLE', 'IOTA', 'ZIL', 'APE', 'STORJ', '1INCH', 'LUNA2', 'FLOW', 'REEF', 'LUNC', 'MINA', 'ASTR', 'ANKR', 'ACH', 'HBAR', 'CKB', 'RDNT', 'HFT', 'ZEC', 'FLOKI', 'SPELL', 'SUI', 'STMX', 'UMA', 'SLP']
lst =   ['ADA', 'LINK', 'DOT', 'ICP', 'AAVE', 'XLM', 'AVAX', 'MANA', 'SAND', 'CRV', 'NEAR', 'PEOPLE', 'ENJ', 'ZIL', 'APE', 'XMR', 'ROSE', 'GAL', 'C98', 'OP', 'RSR', 'LUNA2', 'FLOW', 'LUNC', 'ONE', 'FOOTBALL', 'BAT', 'CELO', 'AGIX', 'ANKR', 'ACH', 'FET', 'RNDR', 'HOOK', 'HBAR', 'COTI', 'VET', 'CKB', 'LIT', 'TLM', 'HOT', 'CHR', 'HFT', 'UNFI', 'DAR', 'SFP', 'SKL', 'CELR', 'SPELL', 'UMA', 'KEY', 'SLP']

print("initial lst:")
print(len(lst))
print(lst)


new_lst = list(dict.fromkeys(lst))

print("no duplicates lst:")
print(len(new_lst))
print(new_lst)

uniqueList = []
duplicateList = []

for i in lst:
    if i not in uniqueList:
        uniqueList.append(i)
    elif i not in duplicateList:
        duplicateList.append(i)

print("duplicates:")
print(len(duplicateList))
print(duplicateList)


for item in new_lst:
    duplicateList.append(item)

print(len(duplicateList))
print(duplicateList)


new_lst = list(dict.fromkeys(duplicateList))

print(len(new_lst))
print(new_lst)


df_new = pd.DataFrame()
df_new['symbol'] = new_lst

df_new.to_csv("../" + "symbols_df_bigwill_usdt.csv", index=False)