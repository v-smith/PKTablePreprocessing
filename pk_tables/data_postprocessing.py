# imports

def split_by_ann(df):
    """Function to return a dictionary of output dfs for each annotator"""
    ann_list= list(df._session_id.unique())
    ann_dfs = {ann.split("-")[3]: df[df['_session_id'] == ann] for ann in ann_list}

    return ann_dfs

def sublist_uniques(data, sublist):
    """Find categories in a column containing lists"""
    categories = set()
    for d, t in data.iterrows():
        try:
            for j in t[sublist]:
                categories.add(j)
        except:
            pass
    return list(categories)


def sublists_to_dummies(f, sublist, index_key=None):
    """Function to convert column containing lists to binary column for each label"""
    categories = sublist_uniques(f, sublist)
    frame = pd.DataFrame(columns=categories)
    for d, i in f.iterrows():
        if type(i[sublist]) == list or np.array:
            try:
                if index_key != None:
                    key = i[index_key]
                    f = np.zeros(len(categories))
                    for j in i[sublist]:
                        f[categories.index(j)] = 1
                    if key in frame.index:
                        for j in i[sublist]:
                            frame.loc[key][j] += 1
                    else:
                        frame.loc[key] = f
                else:
                    f = np.zeros(len(categories))
                    for j in i[sublist]:
                        f[categories.index(j)] = 1
                    frame.loc[d] = f
            except:
                pass

    return frame


