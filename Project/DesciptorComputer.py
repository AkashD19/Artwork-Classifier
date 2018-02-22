# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 19:16:57 2017

@author: dutta
"""

from abc import ABCMeta, abstractmethod

class DescriptorComputer:
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def compute(self, frame):
		pass