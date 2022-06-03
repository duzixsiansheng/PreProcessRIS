#!/usr/bin/python3

import os
from ris import RIS
import json
import traceback
from tornado import web, gen
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote,quote
import cv2
import validators
#from configuration import env

CACHE_ROOT = "/var/www/cache/"
ARCHIVE_ROOT = "/var/www/archive/"

class RisHandler(web.RequestHandler):
    def __init__(self, app, request, **kwargs):
        super(RisHandler, self).__init__(app, request, **kwargs)
        self.executor=ThreadPoolExecutor(4)
	
    # def get(self):
    	
    #     self.write("test")
    #     source_stream=unquote(self.get_argument("source",'00018.jpg'))
    #     ref_stream=unquote(self.get_argument("ref",'15.png'))
    #     style=(unquote(self.get_argument("style",'hair')))
    #     path=(unquote(self.get_argument("path","test")))

    #     self._process(source_stream,ref_stream,style,path)
        #if isinstance(r, dict):
        #print(r)
        #print("done!")
            
        #else:
        #    self.set_status(r[0],r[1])


    def check_origin(self, origin):
        return True

    @run_on_executor
    def _process(self, source, ref, style, path):
        try:
            
            
            print(source)
            RIS_PATH=ARCHIVE_ROOT+"results/"
            if path:
                RIS_PATH=path
            valid_source = validators.url(source)
            valid_ref = validators.url(ref)
            source_stream=source[:-4]
            ref_stream=ref[:-4]
            if (type(source) == str) and ((valid_source == False) and (valid_ref == False)): 
                target_folder=source_stream+"_"+ref_stream
            else:
                target_folder='source_ref'
            target_path=RIS_PATH+target_folder
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            #print(style)
            style_list=style.split(",")
            #print(style_list)
            style_todo=[]
            for item in style_list:
                target_file=os.path.join(target_path,target_folder+"_"+item+".jpg")
                if not os.path.exists(target_file):
                    style_todo.append(item)
            #print(style_todo)
            if len(style_todo) == 0: return {"source":source, "ref":ref, "style":style, "todo": style_todo, "uri": target_path, "status":"done"}
            #print("testsetstest")
            options={
                "input_dir":ARCHIVE_ROOT,
                "output_dir":target_path,
                "im_name1":source,
                "im_name2":ref,
                "display":True,
                "save":True,
                "style":style
            }
            print(options) 
            r = RIS(options) 
            if r:
                return {"source":source, "ref":ref, "style":style, "todo": style_todo, "uri": target_path, "status":"done",'output':r}
            else:
                msg=(503, "Failed in RIS process of RIS service!")
                print(msg,flush=True)
                return msg
        except:
            msg=(503, "Exception in RIS process of RIS service!")
            print(msg,flush=True)
            print(traceback.format_exc(), flush=True)
            return msg

    @gen.coroutine
    def post(self):
        #testsrcimg = cv2.imread('/home/app/input/face/ref/15.png')
        #testrefimg = cv2.imread('/home/app/input/face/source/00018.jpg')

        
        #source_stream=self.get_body_argument("source")
        #ref_stream=self.get_body_argument("ref")
        source_stream=self.get_body_argument("source")
        ref_stream=self.get_body_argument("ref")
        style=(unquote(self.get_argument("style")))
        path=(unquote(self.get_argument("path")))
        print('inputs',source_stream,ref_stream,style,path)
        r=yield self._process(source_stream,ref_stream,style,path)
        if isinstance(r, dict):
            self.write('done!')
            #print(r)
            self.set_header('Content-type','/home/app/testsource_ref/source_reference_hair.jpg')
        else:
            self.set_status(r[0],r[1])
