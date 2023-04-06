# !/usr/bin/python
# coding=utf-8


class Iter():
	'''
	'''
	@staticmethod
	def makeList(x):
		'''Convert the given obj to a list.

		Parameters:
			x () = The object to convert to a list if not already a list, set, or tuple.

		Return:
			(list)
		'''
		return list(x) if isinstance(x, (list, tuple, set, dict, range)) else [x]


	@classmethod
	def nestedDepth(cls, lst, typ=(list, set, tuple)):
		'''Get the maximum nested depth of any sub-lists of the given list.
		If there is nothing nested, 0 will be returned.

		Parameters:
			lst (list): The list to check.
			typ (type)(tuple): The type(s) to include in the query.

		Return:
			(int) 0 if none, else the max nested depth.
		'''
		d=-1
		for i in lst:
			if isinstance(i, typ):
				d = max(cls.nestedDepth(i), d)
		return d+1


	@classmethod
	def flatten(cls, lst):
		'''Flatten arbitrarily nested lists.

		Parameters:
			lst (list): A list with potentially nested lists.

		Return:
			(generator)
		'''
		for i in lst:
			if isinstance(i, (list, tuple, set)):
				for ii in cls.flatten(i):
					yield ii
			else:
				yield i


	@staticmethod
	def collapseList(lst, limit=None, compress=True, toString=True):
		'''Convert a list of integers to a collapsed sequential string format.
		ie. [19,22,23,24,25,26] to ['19', '22..26']

		Parameters:
			lst (list): A list of integers.
			limit (int): limit the maximum length of the returned elements.
			compress (bool): Trim redundant chars from the second half of a compressed set. ie. ['19', '22-32', '1225-6'] from ['19', '22..32', '1225..1226']
			toString (bool): Return a single string value instead of a list.

		Return:
			(list)(str) string if 'toString'.
		'''
		ranges=[]
		for x in map(str, lst): #make sure the list is made up of strings.
			if not ranges:
				ranges.append([x])
			elif int(x)-prev_x==1:
				ranges[-1].append(x)
			else:
				ranges.append([x])
			prev_x = int(x)

		if compress: #style: ['19', '22-32', '1225-6']
			collapsedList = ['-'.join([r[0], r[-1][len(str(r[-1]))-len(str((int(r[-1])-int(r[0])))):]] #find the difference and use that value to further trim redundant chars from the string
								if len(r) > 1 else r) 
									for r in ranges]

		else: #style: ['19', '22..32', '1225..1226']
			collapsedList = ['..'.join([r[0], r[-1]] 
								if len(r) > 1 else r) 
									for r in ranges]

		if limit and len(collapsedList)>limit:
			l = collapsedList[:limit]
			l.append('...')
			collapsedList = l
		
		if toString:
			collapsedList = ', '.join(collapsedList)

		return collapsedList


	@staticmethod
	def bitArrayToList(bitArray):
		'''Convert a binary bitArray to a python list.

		Parameters:
			bitArray () = A bit array or list of bit arrays.

		Return:
			(list) containing values of the indices of the on (True) bits.
		'''
		if len(bitArray):
			if type(bitArray[0])!=bool: #if list of bitArrays: flatten
				lst=[]
				for array in bitArray:
					lst.append([i+1 for i, bit in enumerate(array) if bit==1])
				return [bit for array in lst for bit in array]

			return [i+1 for i, bit in enumerate(bitArray) if bit==1]


	@staticmethod
	def rindex(itr, item):
		'''Get the index of the first item to match the given item 
		starting from the back (right side) of the list.

		Parameters:
			itr (iter): An iterable.
			item () = The item to get the index of.

		Return:
			(int) -1 if element not found.
		'''
		return next(iter(i for i in range(len(itr)-1,-1,-1) if itr[i]==item), -1)


	@staticmethod
	def indices(itr, value):
		'''Get the index of each element of a list matching the given value.

		Parameters:
			itr (iter): An iterable.
			value () = The search value.

		Return:
			(generator)
		'''
		return (i for i, v in enumerate(itr) if v==value)


	@staticmethod
	def removeDuplicates(lst, trailing=True):
		'''Remove all duplicated occurences while keeping the either the first or last.

		Parameters:
			lst (list): The list to remove duplicate elements of.
			trailing (bool): Remove all trailing occurances while keeping the first, else keep last.

		Return:
			(list)
		'''
		if trailing:
			return list(dict.fromkeys(lst))
		else:
			return list(dict.fromkeys(lst[::-1]))[::-1] #reverse the list when removing from the start of the list.


	@classmethod
	def filterDict(cls, dct, inc=[], exc=[], keys=False, values=False):
		'''Filter the given dictionary.
		Extends `filterList` to operate on either the given dict's keys or values.

		Parameters:
			dct (dict): The dictionary to filter.
			inc (str/obj/list): The objects(s) to include.
					supports using the '*' operator: startswith*, *endswith, *contains*
					Will include all items that satisfy ANY of the given search terms.
					meaning: '*.png' and '*Normal*' returns all strings ending in '.png' AND all 
					strings containing 'Normal'. NOT strings satisfying both terms.
			exc (str/obj/list): The objects(s) to exclude. Similar to include.
					exlude take precidence over include.
			keys (bool): Filter the dictionary keys.
			values (bool): Filter the dictionary values.

		Return:
			(dict)

		Example: dct = {1:'1', 'two':2, 3:'three'}
		filterDict(dct, exc='*t*', values=True) #returns: {1: '1', 'two': 2}
		filterDict(dct, exc='t*', keys=True) #returns: {1: '1', 3: 'three'}
		filterDict(dct, exc=1, keys=True) #returns: {'two': 2, 3: 'three'}
		'''
		if keys:
			filtered = cls.filterList(dct.keys(), inc, exc)
			dct = {k: dct[k] for k in filtered}
		if values:
			filtered = cls.filterList(dct.values(), inc, exc)
			dct = {k:v for k,v in dct.items() if v in filtered}
		return dct


	@classmethod
	def filterList(cls, lst, inc=[], exc=[]):
		'''Filter the given list.

		Parameters:
			lst (list): The components(s) to filter.
			inc (str)(int)(obj/list): The objects(s) to include.
					supports using the '*' operator: startswith*, *endswith, *contains*
					Will include all items that satisfy ANY of the given search terms.
					meaning: '*.png' and '*Normal*' returns all strings ending in '.png' AND all 
					strings containing 'Normal'. NOT strings satisfying both terms.
			exc (str)(int)(obj/list): The objects(s) to exclude. Similar to include.
					exlude take precidence over include.
		Return:
			(list)

		Example: filterList([0, 1, 2, 3, 2], [1, 2, 3], 2) #returns: [1, 3]
		'''
		exc = cls.makeList(exc)
		inc = cls.makeList(inc)

		if not any((i for i in inc+exc if isinstance(i, str) and '*' in i)): #if no wildcards used:
			return [i for i in lst if not i in exc and (i in inc if inc else i not in inc)]

		#else: split `inc` and `exc` lists into separate tuples according to wildcard positions. 
		if exc:
			exc_, excContains, excStartsWith, excEndsWith = [],[],[],[]
			for i in exc:
				if isinstance(i, str) and '*' in i:
					if i.startswith('*'):
						if i.endswith('*'):
							excContains.append(i[1:-1])
						excEndsWith.append(i[1:])
					elif i.endswith('*'):
						excStartsWith.append(i[:-1])
				else:
					exc_.append(i)
		if inc:
			inc_, incContains, incStartsWith, incEndsWith = [],[],[],[]
			for i in inc:
				if isinstance(i, str) and '*' in i:
					if i.startswith('*'):
						if i.endswith('*'):
							incContains.append(i[1:-1])
						incEndsWith.append(i[1:])
					elif i.endswith('*'):
						incStartsWith.append(i[:-1])
				else:
					inc_.append(i)

		result=[]
		for i in lst:

			if exc:
				if i in exc_ or isinstance(i, str) and any((
					i.startswith(tuple(excStartsWith)),
					i.endswith(tuple(excEndsWith)),
					next(iter(chars in i for chars in excContains), False))):
					continue

			if inc:
				if i not in inc_ and not (isinstance(i, str) and any(( 
					i.startswith(tuple(incStartsWith)), 
					i.endswith(tuple(incEndsWith)), 
					next(iter(chars in i for chars in incContains), False)))):
					continue

			result.append(i)
		return result


	@staticmethod
	def splitList(lst, into):
		'''Split a list into parts.

		Parameters:
			into (str): Split the list into parts defined by the following:
				'<n>parts' - Split the list into n parts.
					ex. 2 returns:  [[1, 2, 3, 5], [7, 8, 9]] from [1,2,3,5,7,8,9]
				'<n>parts+' - Split the list into n equal parts with any trailing remainder.
					ex. 2 returns:  [[1, 2, 3], [5, 7, 8], [9]] from [1,2,3,5,7,8,9]
				'<n>chunks' - Split into sublists of n size.
					ex. 2 returns: [[1,2], [3,5], [7,8], [9]] from [1,2,3,5,7,8,9]
				'contiguous' - The list will be split by contiguous numerical values.
					ex. 'contiguous' returns: [[1,2,3], [5], [7,8,9]] from [1,2,3,5,7,8,9]
				'range' - The values of 'contiguous' will be limited to the high and low end of each range.
					ex. 'range' returns: [[1,3], [5], [7,9]] from [1,2,3,5,7,8,9]
		Return:
			(list)
		'''
		from string import digits, ascii_letters, punctuation
		mode = into.lower().lstrip(digits)
		digit = into.strip(ascii_letters+punctuation)
		n = int(digit) if digit else None

		if n:
			if mode=='parts':
				n = len(lst)*-1 // n*-1 #ceil
			elif mode=='parts+':
				n = len(lst) // n
			return [lst[i:i+n] for i in range(0, len(lst), n)]

		elif mode=='contiguous' or mode=='range':
			from itertools import groupby
			from operator import itemgetter

			try:
				contiguous = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(lst), lambda x: int(x[0])-int(x[1]))]
			except ValueError as error:
				print ('{} in splitList\n\t# Error: {} #\n\t{}'.format(__file__, error, lst))
				return lst
			if mode=='range':
				return [[i[0], i[-1]] if len(i)>1 else (i) for i in contiguous]
			return contiguous

# --------------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------------

if __name__=='__main__':
	pass

# --------------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------------



# Deprecated ------------------------------------