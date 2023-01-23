# !/usr/bin/python
# coding=utf-8
import json

from pythontk.filetk import File


class Json():
	'''
	'''
	@classmethod
	def setJsonFile(cls, file):
		'''Set the current json filepath.

		:Parameters:
			file (str) = The filepath to a json file. If a file doesn't exist, it will be created.
		'''
		cls._jsonFile = file
		File.getFile(cls._jsonFile) #will create the file if it does not exist.


	@classmethod
	def getJsonFile(cls):
		'''Get the current json filepath.

		:Return:
			(str)
		'''
		try:
			return cls._jsonFile
		except AttributeError as error:
			return ''


	@classmethod
	def setJson(cls, key, value, file=None):
		'''
		:Parameters:
			key () = Set the json key.
			value () = Set the json value for the given key.
			file (str) = Temporarily set the filepath to a json file.
				If no file is given, the previously set file will be used 
				if one was set.

		ex. call: setJson('hdr_map_visibility', state)
		'''
		if not file:
			file = cls.getJsonFile()

		assert file, '{} in setJson\n\t# Error: Operation requires a json file to be specified. #'.format(__file__)
		assert isinstance(file, str), '{} in setJson\n\t# Error:	Incorrect datatype: {} #'.format(__file__, type(file).__name__)

		try:
			with open(file, 'r') as f:
				dct = json.loads(f.read())
				dct[key] = value
		except json.decoder.JSONDecodeError as error:
			dct={}
			dct[key] = value

		with open(file, 'w') as f:
			f.write(json.dumps(dct))


	@classmethod
	def getJson(cls, key, file=None):
		'''
		:Parameters:
			key () = Set the json key.
			value () = Set the json value for the given key.
			file (str) = Temporarily set the filepath to a json file.
					If no file is given, the previously set file will 
					be used if one was set.
		:Return:
			(str)

		ex. call: getJson('hdr_map_visibility') #returns: state
		'''
		if not file:
			file = cls.getJsonFile()

		assert file, '{} in setJson\n\t# Error: Operation requires a json file to be specified. #'.format(__file__)
		assert isinstance(file, str), '{} in setJson\n\t# Error:	Incorrect datatype: {} #'.format(__file__, type(file).__name__)

		try:
			with open(file, 'r') as f:
				return json.loads(f.read())[key]

		except KeyError as error:
			# print ('# Error: {}: getJson: KeyError: {}'.format(__file__, error))
			pass
		except FileNotFoundError as error:
			# print ('# Error: {}: getJson: FileNotFoundError: {}'.format(__file__, error))
			pass
		except json.decoder.JSONDecodeError as error:
			print ('{} in getJson\n\t# Error: JSONDecodeError: {} #'.format(__file__, error))

# --------------------------------------------------------------------------------------------

def __getattr__(attr):
	'''Attempt to get a class attribute.

	:Return:
		(obj)
	'''
	try:
		return getattr(Json, attr)
	except AttributeError as error:
		raise AttributeError(f'{__file__} in __getattr__\n\t{error} ({type(attr).__name__})')







if __name__=='__main__':
	pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------



# Deprecated ------------------------------------