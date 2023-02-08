# !/usr/bin/python
# coding=utf-8
import json

from pythontk.File import getFile


class Json():
	'''
	'''
	@classmethod
	def setJsonFile(cls, file):
		'''Set the current json filepath.

		:Parameters:
			file (str): The filepath to a json file. If a file doesn't exist, it will be created.
		'''
		cls._jsonFile = file
		getFile(cls._jsonFile) #will create the file if it does not exist.


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
			file (str): Temporarily set the filepath to a json file.
				If no file is given, the previously set file will be used 
				if one was set.

		:Example: setJson('hdr_map_visibility', state)
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
			file (str): Temporarily set the filepath to a json file.
					If no file is given, the previously set file will 
					be used if one was set.
		:Return:
			(str)

		:Example: getJson('hdr_map_visibility') #returns: state
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









# --------------------------------------------------------------------------------------------

def __getattr__(attr:str):
	"""Searches for an attribute in this module's classes and returns it.

	:Parameters:
		attr (str): The name of the attribute to search for.
	
	:Return:
		(obj) The found attribute.

	:Raises:
		AttributeError: If the given attribute is not found in any of the classes in the module.
	"""
	try:
		return getattr(Json, attr)

	except AttributeError as error:
		raise AttributeError(f"Module '{__name__}' has no attribute '{attr}'")

# --------------------------------------------------------------------------------------------

if __name__=='__main__':
	pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------



# Deprecated ------------------------------------