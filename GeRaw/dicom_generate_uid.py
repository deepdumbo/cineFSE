# import os
# import numpy
# import string
import commands
def remove_leading_zeros(input_x):
#     output = input
    indx = 0
    while input_x[indx] == '0':
        indx =indx+1
    output = input_x[indx:]
    
    return output
def remove_ending_zeros(input_x):
#     output = input
    indx = -1
    while input_x[indx] == '0':
        indx =indx-1
        
    output = input_x[0:indx+1]
    
    return output
def guid_to_uid(ipt_root,guid_32bit):
    guid = ''
#     print('guid_32bit size',numpy.size(guid_32bit))
    for jj in range(0,len(guid_32bit)):
        tmp="%010.0f"%((guid_32bit[jj]))#.astype(numpy.float128))
        guid=guid+tmp
#         print(guid)
    
#     guid = guid[:12] +guid[13:]
    guid = remove_leading_zeros(guid)
#     guid = remove_ending_zeros(guid)
#     guid = ipt_root + '.2.' + guid
    
    return ipt_root+guid

def dicom_generate_uid(uuid,ipt_root):
    '''
    translate from matlab's dicom_generate_uid
    only linux is supported
    
    '''

#     u = 'ab8ae6cb-0ae4-4f04-ac0a-26200f7a35c0'
#     print('systemuuid=',u)    
    uuid=uuid.replace('-','')
#     guid_32bit= numpy.empty((4))
    guid_32bit = ()

    guid_32bit = guid_32bit +(int(uuid[0:8],16),)
    print((uuid[0:8]))
    print("%010.0f"%int(uuid[0:8],16))
    print("%010.0f"%float(int(uuid[0:8],16)))
    guid_32bit = guid_32bit +(int(uuid[8:16],16),)
    guid_32bit = guid_32bit +(int(uuid[16:24],16),)
    guid_32bit = guid_32bit +(int(uuid[24:32],16),)

#     print(guid_32bit)
#     print('systemuuid=',u)

#     ipt_root='1.2.840.113619.2.5'

    guid = guid_to_uid(ipt_root,guid_32bit)

#     if uid_type == 'instance':
#         pass
#     elif uid_type == 'series':
#         pass
#     elif uid_type == 'study':
#         pass
    return guid
if __name__ == "__main__":
    cmd = 'cat /proc/sys/kernel/random/uuid'
#     t=os.system(cmd)
    uuid = commands.getoutput(cmd)
    ipt_root = '1.3.6.1.4.1.9590.100.1'
    
    guid = dicom_generate_uid(uuid,ipt_root)
    
    print(guid)
#     print(remove_leading_zeros('000sdf00000012e3'))