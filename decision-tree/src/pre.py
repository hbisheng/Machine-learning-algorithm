import sys
import os
from ds import entry
from tools import validate_entry

def load_attribute(attribute_path):
    #load the attribute values from adult names
    #a parser for .name file
    if not os.path.exists(attribute_path):
        print "Invalid Attribute Path"
        return None
    attr_file = open(attribute_path,'rb');
    attr_content = attr_file.readlines()
    attr_file.close()
    
    label = []
    attr_list = []
    for line in attr_content:
        if line.startswith('|') or len(line) == 0 or line.isspace():
            continue
        line = line.replace('.',' ')
        line = line.replace('\n','')
        column = line.find(':')
        if column != -1: # this is a attr
            attr_name = line[0 : column].strip()
            attr_value = []
            values = line[column + 1:].split(',')
            for v in values:            
                attr_value.append(v.strip())  
            attr = {}
            attr['name'] = attr_name
            attr['values'] = attr_value
            attr_list.append(attr)
        else: # this is a label
            content = line.split(',')
            for c in content:
                label.append(c.strip())
    if label == None:
        print 'No label Found'
    return (label, attr_list)                     
        
    
def load_data(attributes, data_path): 
    #attributes are for creating entries
    if not os.path.exists(data_path):
        print "Invalid Attribute Path"
        return None
    data_file = open(data_path, 'rb')
    data_content = data_file.readlines()
    data_file.close()
    entry_list = []
    for line in data_content:
        line = line.replace('.',' ')
        line = line.replace('\n','')
        e = entry(attributes, line.split(','))
        if validate_entry(attributes, e):
            entry_list.append(e)
        else:
            print "Bad Entry Found!"        
    return entry_list
        
    
    