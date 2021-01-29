import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from pathlib import Path
import pandas as pd
from intervals import FloatInterval
import intervals
import numpy as np


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

def print_summary_by_subject(df):
    maxval_len = max(df['Length'])
    maxval_ser = max(df['SeriesNo'])
    @interact
    def show_series(ID='Z1243452'):
        df_tmp = None
        df_tmp = df.loc[df['ID'] == ID]
        print('No of Series: ', len(df_tmp))
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


# Functions for Pair Validity

def remove_series(imseries,serieslist,exlist):
    if imseries in serieslist:
        serieslist.remove(imseries)
    if imseries not in exlist:
        exlist.append(imseries)
    return serieslist,exlist

def get_seriesID(df,ID):
    return df.loc[df['ID']==ID,'SeriesID'].values.tolist()
    
def calculate_series_overlap(series1,series2):
    try:
        interval1 = FloatInterval(
        [
            round(np.min(series1.z_range_pair)),
            round(np.max(series1.z_range_pair)),
        ]
        )
        interval2 = FloatInterval(
            [
                round(np.min(series2.z_range_pair)),
                round(np.max(series2.z_range_pair)),
            ]
        )

        overlap = interval1 & interval2
        longer_length = max(interval1.length,interval2.length)
        return round((overlap.length / longer_length),3)
    except intervals.exc.IllegalArgument: # raised if there is no overlap
        return False
    except Exception as e:
        return e


def calculate_missing_slices_axials(series1):
    try:
        interval1 = FloatInterval(
            [
                round(np.min(series1.z_range_pair)),
                round(np.max(series1.z_range_pair)),
            ]
        )
        
        Thickness_1 = series1.slice_thickness
        nslices1 =  round(interval1.length/Thickness_1)
        missingslices1 = nslices1 - series1.number_of_dicoms
        if missingslices1 < 0:
            return 1.0
        else:
            missinglength1 = missingslices1*Thickness_1

            # print('nslices: ',nslices1, ' interval_len: ', interval1.length, 
            #       'missing slices: ', missingslices1, 'missing length: ',missinglength1)

            return  round(1 - (missinglength1)/(interval1.length),3)
    except:
        return 0


def calculate_missing_slices_sagittals(series2):
    try:
        interval2 = FloatInterval(
            [
                round(min(series2.slice_loc[0][0],series2.slice_loc[-1][0])),
                round(max(series2.slice_loc[0][0],series2.slice_loc[-1][0])),
            ]
        )
        
        Thickness_2 = series2.slice_thickness
        nslices2 =  round(interval2.length/Thickness_2)
        missingslices2 = nslices2 - series2.number_of_dicoms
        if missingslices2 < 0:
            missinglength2 = 0
        else:
            missinglength2 = missingslices2*Thickness_2
        # print('nslices: ',nslices2, ' interval_len: ', interval2.length, 'missing slices: ', missingslices2, 'missing length: ',missinglength2)
        return  round(1 - (missinglength2)/(interval2.length),3)
    except:
        return 0
        
def calculate_missing_slices_score(series1,series2):
    try:
        interval1 = FloatInterval(
            [
                round(np.min(series1.z_range_pair)),
                round(np.max(series1.z_range_pair)),
            ]
        )
        Thickness_1 = series1.slice_thickness
        nslices1 =  round(interval1.length/Thickness_1)
        missingslices1 = nslices1 - series1.number_of_dicoms
        
        if missingslices1 < 0:
            missinglength1 = 0
        else:
            missinglength1 = missingslices1*Thickness_1
    except:
        return 0

    try:
        interval2 = FloatInterval(
            [
                round(min(series2.slice_loc[0][0],series2.slice_loc[-1][0])),
                round(max(series2.slice_loc[0][0],series2.slice_loc[-1][0])),
            ]
        )
        Thickness_2 = series2.slice_thickness
        nslices2 =  round(interval2.length/Thickness_2)
        missingslices2 = nslices2 - series2.number_of_dicoms
        if missingslices2 < 0:
            missinglength2 = 0
        else:
            missinglength2 = missingslices2*Thickness_2
    except:
        return 0    
    #print("int1: ", interval1, "int2: ", interval2)
    return  round(1 - (missinglength1+missinglength2)/(interval1.length+interval2.length),3)
    
    
def print_series_overlap(df_ax,df_sag,axial_series,sagittal_series,subjects):
    @interact
    def show_overlap(ID = 'Z416634'):
        df  = pd.DataFrame(columns=['Axial','Sagittal','Overlap','MissingScore','PairValidity','AxSlices','SagSlices'])
        axials = get_seriesID(df_ax,ID)
        sagittals = get_seriesID(df_sag,ID)
        axseries = []
        sagseries = []
        for axial in axials:
            axseries.append(get_subject_series(ID,axial,subjects))
        for sag in sagittals:
            sagseries.append(get_subject_series(ID,sag,subjects))
        
        i = 0   
        for a in axseries:
            for s in sagseries:
                df.loc[i,'AxSlices'] = a.number_of_dicoms
                df.loc[i,'Axial'] = a.id_
                df.loc[i,'SagSlices'] = s.number_of_dicoms
                df.loc[i,'Sagittal'] = s.id_
                df.loc[i,'Overlap'] = round(calculate_series_overlap(a,s),3)
                df.loc[i,'MissingScore'] = round(calculate_missing_slices_score(a,s),3)
                df.loc[i,'PairValidity'] = df.loc[i,'Overlap'] + df.loc[i,'MissingScore']
                i += 1

        return df.sort_values(by=['PairValidity'], ascending=False)
    

# MOre sophisticated than get_finalpairs_df, used for incomplete studies & Parallel implementation that returns lists    
def filter_finalpairs(ID,df_ax,df_sag,subjects):
    try:
        axials = get_seriesID(df_ax,ID)
        sagittals = get_seriesID(df_sag,ID)
        axseries = []
        sagseries = []
        for axial in axials:
            a_s = get_subject_series(ID,axial,subjects)
            if a_s.number_of_dicoms > 10:
                axseries.append(a_s)
        for sag in sagittals:
            s_s = get_subject_series(ID,sag,subjects)
            if s_s.number_of_dicoms > 10:
                sagseries.append(s_s)

        i = 0
        df_tmp =  pd.DataFrame(columns=['Axial','Sagittal','Overlap','MissingScore','PairValidity','AxSlices','SagSlices','AxThick','SagThick'])
        if axseries: 
            for a in axseries:
                df_tmp.loc[i,:] = None
                if sagseries:
                    for s in sagseries:
                        df_tmp.loc[i,'AxSlices'] = a.number_of_dicoms
                        df_tmp.loc[i,'SagSlices'] = s.number_of_dicoms
                        df_tmp.loc[i,'Axial'] = a.id_
                        df_tmp.loc[i,'Sagittal'] = s.id_
                        df_tmp.loc[i,'Overlap'] = calculate_series_overlap(a,s)
                        df_tmp.loc[i,'MissingScore'] = calculate_missing_slices_score(a,s)
                        if isinstance(df_tmp.loc[i,'Overlap'], (int,float)):
                            df_tmp.loc[i,'PairValidity'] = df_tmp.loc[i,'Overlap'] + df_tmp.loc[i,'MissingScore']
                        else:
                            df_tmp.loc[i,'PairValidity'] =  df_tmp.loc[i,'MissingScore'] 
                        
                        try:
                            df_tmp.loc[i,'AxThick'] = a.slice_thickness
                        except:
                            df_tmp.loc[i,'AxThick'] = 0
                        
                        try:
                            df_tmp.loc[i,'SagThick'] = s.slice_thickness
                        except:
                            df_tmp.loc[i,'SagThick'] = 0

                        i += 1
                else:
                        df_tmp.loc[i,'AxSlices'] = a.number_of_dicoms
                        df_tmp.loc[i,'Axial'] = a.id_
                        try:
                            df_tmp.loc[i,'AxThick'] = a.slice_thickness
                        except:
                            df_tmp.loc[i,'AxThick'] = 0
                        
                        df_tmp.loc[i,'MissingScore'] = round(calculate_missing_slices_axials(a),3)
                        i += 1
        elif sagseries:  
            for s in sagseries:
                df_tmp.loc[i,:] = None
                df_tmp.loc[i,'SagSlices'] = s.number_of_dicoms
                df_tmp.loc[i,'Sagittal'] = s.id_
                df_tmp.loc[i,'MissingScore'] = round(calculate_missing_slices_sagittals(s),3)
                try:
                    df_tmp.loc[i,'SagThick'] = s.slice_thickness
                except:
                    df_tmp.loc[i,'SagThick'] = 0
                    
                i += 1
        
        df_tmp =  df_tmp.sort_values(by=['PairValidity','MissingScore','AxThick','SagThick','SagSlices'], 
                                     ascending=[False,False,False,False,True])
        if axseries or sagseries:
            #display(df_tmp)
            result = df_tmp.iloc[0,:].values.tolist()
            result.insert(0,ID)
            return result
    except Exception as e:
        return [ID,None,None,None,None,None,None,None,None,None]



def get_finalpairs_df(df_ax,df_sag,subjects):
    df  = pd.DataFrame(columns=['ID','Axial','Sagittal','Overlap','MissingScore','PairValidity','AxSlices','SagSlices'])
    
    for sid,subject in enumerate(subjects):
        ID = subject.id_
        print('Processing: ',sid,"  ", ID) 
        try:
            axials = get_seriesID(df_ax,ID)
            sagittals = get_seriesID(df_sag,ID)
            axseries = []
            sagseries = []
            for axial in axials:
                axseries.append(get_subject_series(ID,axial,subjects))
            for sag in sagittals:
                sagseries.append(get_subject_series(ID,sag,subjects))

            df.loc[sid,'ID'] = ID
            df.iloc[sid,1:] = None
            i = 0
            df_tmp =  pd.DataFrame(columns=['Axial','Sagittal','Overlap','MissingScore','PairValidity','AxSlices','SagSlices'])
            if axseries: 
                for a in axseries:
                    df_tmp.loc[i,:] = None
                    if sagseries:
                        for s in sagseries:
                            df_tmp.loc[i,'AxSlices'] = a.number_of_dicoms
                            df_tmp.loc[i,'SagSlices'] = s.number_of_dicoms
                            df_tmp.loc[i,'Axial'] = a.id_
                            df_tmp.loc[i,'Sagittal'] = s.id_
                            df_tmp.loc[i,'Overlap'] = round(calculate_series_overlap(a,s),3)
                            df_tmp.loc[i,'MissingScore'] = round(calculate_missing_slices_score(a,s),3)
                            df_tmp.loc[i,'PairValidity'] = df_tmp.loc[i,'Overlap'] + df_tmp.loc[i,'MissingScore']
                            i += 1
                    else:
                            df_tmp.loc[i,'AxSlices'] = a.number_of_dicoms
                            df_tmp.loc[i,'Axial'] = a.id_
                            df_tmp.loc[i,'MissingScore'] = round(calculate_missing_slices_axials(a),3)
                            i += 1
            elif sagseries:  
                for s in sagseries:
                    df_tmp.loc[i,:] = None
                    df_tmp.loc[i,'SagSlices'] = s.number_of_dicoms
                    df_tmp.loc[i,'Sagittal'] = s.id_
                    df_tmp.loc[i,'MissingScore'] = round(calculate_missing_slices_sagittals(s),3)
                    i += 1


            df_tmp =  df_tmp.sort_values(by=['PairValidity','MissingScore'], ascending=False)
            if axseries or sagseries:
                df.iloc[sid,1:] = df_tmp.iloc[0,:]
        except Exception as e:
            print('Error: ',str(e))
            

    return df


# Function to Exlude Subjects from subject and series lists:

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

# Remove subjects
def exclude_subjects(subs,axseries,sagseries,exsubjects,ID):
    # Remove in subjects
    try:
        index = [x.id_ for x in subs].index(ID)
    except:
        index = None
    if index: 
        exsubjects.append(subs[index])
        del subs[index]
        
    # Remove in Axial series
    axlist = [x.subject.id_ for x in axseries]
    try:
        index = indices(axlist,ID)
    except:
        index = None
    if index:
        axseries= [i for j, i in enumerate(axseries) if j not in index]
    
    # Remove in Sagittal series
    saglist = [x.subject.id_ for x in sagseries]
    try:
        index = indices(saglist,ID)
    except:
        index = None
    if index:
        sagseries= [i for j, i in enumerate(sagseries) if j not in index]

    return (subs,axseries,sagseries,exsubjects)