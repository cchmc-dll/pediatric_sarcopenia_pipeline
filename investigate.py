import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from pathlib import Path
import pandas as pd


## Functions to remove duplicates in Section 2

# Find all series paths for a subject
def print_subject_paths(subjects):
    paths = None
    @interact
    def show(ID='Z1243452'):
        paths =  [str(subject.path) for subject in subjects if subject.id_==ID]
        [print(i, ' ',path) for i,path in enumerate(paths)]

def print_subject_series(v1,v2,subjects,targetseries):
    @interact
    def show(ID=v1,path=v2):
        for subject in subjects:
            if subject.id_==ID:
               # print('Subject ', ID, ' found')
                if str(subject.path) == path:
                #    print('Subject path', path, ' found')
                    seriess =  list(subject.find_series())
                    [print(i,' ', str(s.id_)) for i,s in enumerate(seriess) if s in targetseries]
                    
def get_subject_series(ID,SID,subjects):
    for subject in subjects:
        if subject.id_==ID:
            seriess =  list(subject.find_series())
            for s in (seriess):
                if s.id_==SID:
                    return s 

def get_subject_series(ID,SID,subjects):
    for subject in subjects:
        if subject.id_==ID:
            seriess =  list(subject.find_series())
            for s in (seriess):
                if s.id_==SID:
                    return s 


# Section 3 - Process Series functions
def get_summary_dfs(axial_series,sagittal_series,subjects):
    df  = pd.DataFrame(columns=['ID','Axials','Sagittals'])
    for i,subject in enumerate(subjects):
        df.loc[i,'ID'] = subject.id_
        df.loc[i,'Axials'] = 0
        df.loc[i,'Sagittals'] = 0

    for series in axial_series:
        patid = series.subject.id_
        df.loc[df['ID']==patid,'Axials'] += 1

    for series in sagittal_series:
        patid = series.subject.id_
        df.loc[df['ID']==patid,'Sagittals'] += 1
    
    return df


def get_summary_by_serieslength(serieslist):
    df  = pd.DataFrame(columns=['ID','SeriesNo','Length','Thickness','SeriesID'])
    tmpdict = {}
    for i,series in enumerate(serieslist):
        df.loc[i,'ID'] = series.subject.id_
        if not series.subject.id_ in tmpdict:
            tmpdict[series.subject.id_] = 1
        else:
            tmpdict[series.subject.id_] += 1

        df.loc[i,'SeriesNo'] = tmpdict[series.subject.id_]
        df.loc[i,'Length'] = series.number_of_dicoms
        df.loc[i,'SeriesID'] = series.id_
        if hasattr(series,'slice_thickness'):
            df.loc[i,'Thickness'] = series.slice_thickness

    return df.sort_values(by=['ID','Length'],ascending=False)
 
def print_summary_by_serieslength(df):
    maxval_len = max(df['Length'])
    maxval_ser = max(df['SeriesNo'])
    @interact
    def show_series(ptype = ['equal','equal_greater','lesser'], length=(0,maxval_len,1), nseries=(0,maxval_ser,1),sorting=['ascending','descending']):
        df_tmp = None
        if ptype == 'equal':
            df_tmp = df.loc[(df['Length'] == length) & (df['SeriesNo'] >= nseries)]           
        elif ptype == 'equal_greater':
            df_tmp = df.loc[(df['Length'] >= length) & (df['SeriesNo'] >= nseries)]
        else:
            df_tmp = df.loc[(df['Length'] < length) & (df['SeriesNo'] >= nseries)]
    
        print('No of Subjects: ', len(df_tmp['ID'].unique()))
        print('No of Series: ', len(df_tmp))
        if sorting == 'ascending':
            return df_tmp.sort_values(by=['SeriesNo','ID','Length'],ascending=True)
        else:
            return df_tmp.sort_values(by=['SeriesNo','ID','Length'],ascending=False)



def get_patientsbycount(df,column,count):
    return df[df[column] == count]['ID'].values.tolist() 

def print_summary_df(df):
    maxval = max(df[["Axials", "Sagittals"]].max(axis=1))
    @interact
    def show_series(column=['Axials','Sagittals'], x=(0,maxval,1)):
        df_tmp = df.loc[df[column] == x]
        print('Subjects: ', len(df_tmp))
        return df_tmp.sort_values(by=column,ascending=False)

def print_summary_counts(df):
    maxval = max(df[["Axials", "Sagittals"]].max(axis=1))
    @interact
    def show_count(column=['Axials','Sagittals'], x=(0,maxval,1)):
        print(column,  ' patient count')
        for i in range(int(x)):
            print(i, '             ',  len(df[df[column] == i]))
            


## Section 4: Functions to investigate filtered series
def get_exclusion_df(exclusions):
    df  = pd.DataFrame(columns=['index','ID','SeriesType','Reason','SliceThickness','Name'])
    for i,iseries in enumerate(exclusions):
        df.loc[i,'index'] = i
        df.loc[i,'ID'] = iseries[0].series.subject.id_
        df.loc[i,'SeriesType'] = iseries[0].series.orientation
        df.loc[i,'Reason'] = iseries[0].reason
        df.loc[i,'SliceThickness'] = str(iseries[0].series.slice_thickness)
        df.loc[i,'Name'] = iseries[0].series.id_
    
    df.set_index('index')
    return df

def get_exclusionpath(idx,exclusions):
    return exclusions[idx][0].series.series_path
            
def print_summary_exclusions(exclusions):
    reasons = [ex[0].reason for ex in exclusions]
    ureasons = set(reasons)
    @interact
    def show_reasons(x=(0,2000,50)):
        print(" Reason      Count")
        for reason in ureasons:
            c = reasons.count(reason)
            if c > x:
                print(reason, "     ", reasons.count(reason)) 

def print_exclusions_subject(df,ID):
    @interact
    def show():
        df_tmp = df.loc[df['ID'].isin(ID)]
        print(" Total number of series excluded for patient list is ", len(df_tmp))
        return df_tmp