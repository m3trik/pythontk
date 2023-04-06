# !/usr/bin/python
# coding=utf-8
import sys, os, traceback
#from this package:
from pythontk._core import Core
from pythontk._iter import Iter


class File():
	'''
	'''
	@staticmethod
	def getFile(filepath, mode='a+'):
		'''Return a file object with the given mode.

		Parameters:
			filepath (str): The path to an existing file or the desired location for one to be created.
			mode (str): 'r' - Read - Default value. Opens a file for reading, error if the file does not exist.
				'a' - Append - Opens a file for appending, creates the file if it does not exist.
				'a+' - Read+Write - Creates a new file or opens an existing file, the file pointer position at the end of the file.
				'w' - Write - Opens a file for writing, creates the file if it does not exist.
				'w+' - Read+Write - Opens a file for reading and writing, creates the file if it does not exist.
				'x' - Create - Creates a new file, returns an error if the file exists.
				't' - Text - Default value. Text mode
				'b' - Binary - Binary mode (e.g. images)

		Return:
			(obj) file
		'''
		try:
			with open(filepath, mode) as f:
				return f
		except OSError as error:
			traceback.print_exc()


	@staticmethod
	def getFileContents(filepath: str, asList=False) -> None:
		'''Get each line of a text file as indices of a list.
		Will create a file if one doesn't already exist.

		Parameters:
			filepath (str): The path to an existing text based file.
			asList (bool): Return as a list or a string.

		Return:
			(list)
		'''
		try:
			with open(filepath, 'r') as f:
				return f.readlines() if asList else f.read()
		except OSError as error:
			traceback.print_exc()


	@staticmethod
	def writeToFile(filepath, lines):
		'''Write the given list contents to the given file.

		Parameters:
			filepath (str): The path to an existing text based file.
			lines (list): A list of strings to write to the file.
		'''
		try:
			with open(filepath, 'w') as f:
				f.writelines(lines)
		except OSError as error:
			traceback.print_exc()


	@staticmethod
	@Core.listify
	def formatPath(p, section='', replace=''):
		'''Format a given filepath(s).
		When a section arg is given, the correlating section of the string will be returned.
		If a replace arg is given, the stated section will be replaced by the given value.

		Parameters:
			p (str/list): The filepath(s) to be formatted.
			section (str): The desired subsection of the given path. 
					'path' path - filename, 
					'dir'  directory name, 
					'file' filename + ext, 
					'name', filename - ext,
					'ext', file extension,
					(if '' is given, the fullpath will be returned)
		Return:
			(str/list) List if 'strings' given as list.
		'''
		if not isinstance(p, (str)):
			return p

		p = os.path.expandvars(p) #convert any env variables to their values.
		p = p[:2]+'/'.join(p[2:].split('\\')).rstrip('/') #convert forward slashes to back slashes.

		fullpath = p if '/' in p else ''
		filename_ = p.split('/')[-1]
		filename = filename_ if '.' in filename_ and not filename_.startswith('.') else ''
		path = '/'.join(p.split('/')[:-1]) if filename else p
		directory = p.split('/')[-2] if (filename and path) else p.split('/')[-1]
		name = ''.join(filename.rsplit('.', 1)[:-1]) if filename else '' if fullpath else p
		ext = filename.rsplit('.', 1)[-1]

		orig_str = p #the full original string (formatted with forwardslashes)

		if section=='path':
			p = path

		elif section=='dir':
			p = directory

		elif section=='file':
			p = filename

		elif section=='name':
			p = name

		elif section=='ext':
			p = ext

		if replace:
			p = Str_utils.rreplace(orig_str, p, replace, 1)

		return p


	@classmethod
	def timeStamp(cls, filepaths, detach=False, stamp='%m-%d-%Y  %H:%M', sort=False):
		'''Attach a modified timestamp and date to given file path(s).

		Parameters:
			filepaths (str/list): The full path to a file. ie. 'C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'
			detach (bool): Remove a previously attached time stamp.
			stamp (str): The time stamp format.
			sort (bool): Reorder the list of filepaths by time. (most recent first)

		Return:
			(list) ie. ['16:46  11-09-2021  C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb'] from ['C:/Windows/Temp/__AUTO-SAVE__untitled.0001.mb']
		'''
		from datetime import datetime
		import os.path

		files = cls.formatPath(Iter.makeList(filepaths))

		result=[]
		if detach:
			for f in files:
				if len(f)>2 and not any(['/'==f[2], '\\'==f[2], '\\\\'==f[:2]]): #attempt to decipher whether the path has a time stamp.
					strip = ''.join(f.split()[2:])
					result.append(strip)
				else:
					result.append(f)
		else:
			for f in files:
				try:
					result.append('{}  {}'.format(datetime.fromtimestamp(os.path.getmtime(f)).strftime(stamp), f))
				except (FileNotFoundError, OSError) as error:
					continue

			if sort:
				result = list(reversed(sorted(result)))

		return result


	@staticmethod
	def isValidPath(filepath: str) -> list:
		'''Determine if the given filepath is valid.

		Parameters:
			filepath (str): The path to a file.

		Return:
			(str) The path type (ie. 'file' or 'dir') or None.
		'''
		fp = os.path.expandvars(filepath) #convert any env variables to their values.

		if os.path.isfile(fp):
			return 'file'
		elif os.path.isdir(fp):
			return 'dir'
		return None


	@staticmethod
	def createDir(filepath: str) -> None:
		'''Create a directory if one doesn't already exist.

		Parameters:
			filepath (str): The path to where the file will be created.
		'''
		fp = os.path.expandvars(filepath) #convert any env variables to their values.
		try:
			if not os.path.exists(fp):
				os.makedirs(fp)
		except OSError as error:
			print ('{} in createDir\n\t# Error: {}.\n\tConfirm that the following path is correct: #\n\t{}'.format(__file__, error, fp))


	@classmethod
	def getDirContents(cls, path, returnType='files', recursive=False, topdown=True, reverse=False, 
								incFiles=[], excFiles=[], incDirs=[], excDirs=[]):
		'''Get the contents of a directory and any of it's children.

		Parameters:
			path (str): The path to the directory.
			returnType (str): Return files and directories. Multiple types can be given using '|' 
					ex. 'files|dirs' (valid: 'files'(default), filenames, 'filepaths', 'dirs', 'dirpaths')
					case insensitive. singular or plural.
			recursive (bool): return the contents of the root dir only.
			topDown (bool): Scan directories from the top-down, or bottom-up.
			reverse (bool): When True, reverse the final result.
			incFiles (str/list): Include only specific files.
			excFiles (str/list): Excluded specific files.
			incDirs (str/list): Include only specific child directories.
			excDirs (str/list): Excluded specific child directories.
					supports using the '*' operator: startswith*, *endswith, *contains*
					ex. *.ext will exclude all files with the given extension.
					exclude takes precedence over include.
		Return:
			(list)

		ex. getDirContents(path, returnType='filepaths')
		ex. getDirContents(path, returnType='files|dirs')
		'''
		path = os.path.expandvars(path) #translate any system variables that might have been used in the path.
		returnTypes = [t.strip().rstrip('s').lower() for t in returnType.split('|')] #strip any whitespace and trailing 's' of the types to allow for singular and plural to be used interchagably. ie. files | dirs becomes [file, dir]

		result=[]
		for root, dirs, files in os.walk(path, topdown=topdown):
			dirs[:] = Iter.filterList(dirs, incDirs, excDirs) #remove any directories in 'exclude'.

			if 'dir' in returnTypes:
				for d in dirs:
					result.append(d) #get the dir contents before filtering for root.
			if 'dirpath' in returnTypes:
				for d in dirs:
					result.append(os.path.join(root, d))
			if not recursive:
				dirs[:] = [d for d in dirs if d is root] #remove all but the root dir.

			files[:] = Iter.filterList(files, incFiles, excFiles) #remove any files in 'exclude'.
			for f in files:
				if 'file' in returnTypes:
					result.append(f)
				if 'filename' in returnTypes:
					n = cls.formatPath(f, 'name')
					result.append(n)
				if 'filepath' in returnTypes:
					result.append(os.path.join(root, f))
		if reverse:
			return result[::-1]
		return result


	@staticmethod
	def getFilepath(obj, incFilename=False):
		'''Get the filepath of a class or module.

		Parameters:
			obj (obj): A python module, class, or the built-in __file__ variable.
			incFilename (bool): Include the filename in the returned result.

		Return:
			(str)
		'''
		from types import ModuleType

		if isinstance(obj, type(None)):
			return ''
		elif isinstance(obj, str):
			filepath = obj
		elif isinstance(obj, ModuleType):
			filepath = obj.__file__
		else:
			clss = obj if callable(obj) else obj.__class__
			try:
				import inspect
				filepath = inspect.getfile(clss)

			except TypeError as error: #iterate over each filepath in the call frames, until a class with a matching name is found.
				import importlib

				filepath=''
				for frame_record in inspect.stack():
					if filepath:
						break
					frame = frame_record[0]
					_filepath = inspect.getframeinfo(frame).filename
					mod_name = os.path.splitext(os.path.basename(_filepath))[0]
					spec = importlib.util.spec_from_file_location(mod_name, _filepath)
					if not spec:
						continue
					mod = importlib.util.module_from_spec(spec)
					spec.loader.exec_module(mod)

					for cls_name, clss_ in inspect.getmembers(mod, inspect.isclass): #get the module's classes.
						if cls_name==clss.__name__:
							filepath = _filepath
		if incFilename:
			return os.path.abspath(filepath)
		else:
			return os.path.abspath(os.path.dirname(filepath))


	@staticmethod
	def appendPaths(rootDir, ignoreStartingWith=('.', '__'), verbose=False):
		'''Append all sub-directories of the given 'rootDir' to the python path.

		Parameters:
			rootDir (str): Sub-directories of this directory will be appended to the system path.
			ignoreStartingWith (str)(tuple): Ignore directories starting with the given chars.
			verbose (bool): Output the results to the console. (Debug)
		'''
		path = os.path.dirname(os.path.abspath(rootDir))
		sys.path.insert(0, path)
		if verbose:
			print (path)

		# recursively append subdirectories to the system path.
		for root, dirs, files in os.walk(path):
			dirs[:] = [d for d in dirs if not d.startswith(ignoreStartingWith)]
			for dir_name in dirs:
				dir_path = os.path.join(root, dir_name)
				sys.path.insert(0, dir_path)
				if verbose:
					print (dir_path)


	@classmethod
	def updateVersion(cls, filepath: str, change: str='increment', version_part: str='patch', max_version_parts: tuple=(9, 9)) -> None:
		"""This function updates the version number in a text file depending on its state.
		The version number is defined as a line in the following format: __version__ = "0.0.0"

		The version number is represented as a string in the format 'x.y.z', where x, y, and z are integers. 

		Parameters:
			filepath (str): The path to the text file containing the version number.
			change (str, optional): The type of change, either 'increment' or 'decrement'. Defaults to 'increment'.
			version_part (str, optional): The part of the version number to update, either 'major', 'minor', or 'patch'. Defaults to 'patch'.
			max_version_parts (tuple, optional): A tuple containing the maximum values for the minor and patch version parts. Defaults to (9, 9).

		Return:
			(str): The new version number.
		"""
		import re

		lines = cls.getFileContents(filepath, asList=True)

		version_pattern = re.compile(r"__version__\s*=\s*['\"](\d+)\.(\d+)\.(\d+)['\"]")
		max_minor, max_patch = max_version_parts

		version = ''
		for i, line in enumerate(lines):
			match = version_pattern.match(line)
			if match:
				major, minor, patch = map(int, match.groups())

				if version_part == 'patch':
					if change == 'increment':
						patch = (patch + 1) % (max_patch + 1)
						if patch == 0:
							minor = (minor + 1) % (max_minor + 1)
							major += minor == 0
					elif change == 'decrement':
						if patch == 0:
							patch = max_patch
							minor = (minor - 1) % (max_minor + 1) if minor > 0 else max_minor
							major -= minor == max_minor
						else:
							patch -= 1
					else:
						raise ValueError("Invalid change parameter. Use either 'increment' or 'decrement'.")
				elif version_part == 'minor':
					if change == 'increment':
						minor = (minor + 1) % (max_minor + 1)
						major += minor == 0
					elif change == 'decrement':
						minor = (minor - 1) % (max_minor + 1)
					else:
						raise ValueError("Invalid change parameter. Use either 'increment' or 'decrement'.")
				elif version_part == 'major':
					if change == 'increment':
						major += 1
					elif change == 'decrement':
						major = max(0, major - 1)
					else:
						raise ValueError("Invalid change parameter. Use either 'increment' or 'decrement'.")
				else:
					raise ValueError("Invalid version_part parameter. Use either 'major', 'minor', or 'patch'.")

				version = f"{major}.{minor}.{patch}"
				lines[i] = f"__version__ = '{version}'\n"
				break

		cls.writeToFile(filepath, lines)
		if not version:
			print (f"# Error: No version in the format: __version__ = \"0.0.0\" found in {filepath}")
		return version

# --------------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------------

if __name__=='__main__':
	pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------



# Deprecated ------------------------------------


	# def getCallingModuleDir():
	# 	"""Get the directory path of the module that called the function.

	# 	Return:
	# 		(str) The directory path of the calling module.
	# 	"""
	# 	import os, inspect

	# 	calling_frame = inspect.currentframe().f_back
	# 	calling_module = inspect.getmodule(calling_frame)
	# 	calling_module_path = os.path.abspath(calling_module.__file__)
	# 	calling_module_dir = os.path.dirname(calling_module_path)

	# 	return calling_module_dir