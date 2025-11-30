# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'map_packer.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,
    QLabel, QLayout, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QTabWidget, QVBoxLayout,
    QWidget)

from uitk.widgets.comboBox.ComboBox import ComboBox
from uitk.widgets.header.Header import Header
from widgets.lineedit import LineEdit

class Ui_QtUi(object):
    def setupUi(self, QtUi):
        if not QtUi.objectName():
            QtUi.setObjectName(u"QtUi")
        QtUi.setEnabled(True)
        QtUi.resize(200, 307)
        QtUi.setTabShape(QTabWidget.Triangular)
        QtUi.setDockNestingEnabled(True)
        QtUi.setDockOptions(QMainWindow.AllowNestedDocks|QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks|QMainWindow.ForceTabbedDocks)
        self.central_widget = QWidget(QtUi)
        self.central_widget.setObjectName(u"central_widget")
        self.central_widget.setMinimumSize(QSize(200, 0))
        self.verticalLayout = QVBoxLayout(self.central_widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(6)
        self.main_layout.setObjectName(u"main_layout")
        self.main_layout.setSizeConstraint(QLayout.SetFixedSize)
        self.header = Header(self.central_widget)
        self.header.setObjectName(u"header")
        self.header.setMinimumSize(QSize(0, 22))
        self.header.setMaximumSize(QSize(999, 22))
        font = QFont()
        font.setBold(True)
        self.header.setFont(font)

        self.main_layout.addWidget(self.header)

        self.Formatting = QGroupBox(self.central_widget)
        self.Formatting.setObjectName(u"Formatting")
        self.gridLayout = QGridLayout(self.Formatting)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.cmbB = ComboBox(self.Formatting)
        self.cmbB.setObjectName(u"cmbB")
        self.cmbB.setMinimumSize(QSize(0, 20))
        self.cmbB.setMaximumSize(QSize(16777215, 20))
        self.cmbB.setMaxVisibleItems(30)
        self.cmbB.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.cmbB.setFrame(False)

        self.gridLayout.addWidget(self.cmbB, 2, 1, 1, 1)

        self.cmbR = ComboBox(self.Formatting)
        self.cmbR.setObjectName(u"cmbR")
        self.cmbR.setMinimumSize(QSize(0, 20))
        self.cmbR.setMaximumSize(QSize(16777215, 20))
        self.cmbR.setMaxVisibleItems(30)
        self.cmbR.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.cmbR.setFrame(False)

        self.gridLayout.addWidget(self.cmbR, 0, 1, 1, 1)

        self.cmbA = ComboBox(self.Formatting)
        self.cmbA.setObjectName(u"cmbA")
        self.cmbA.setMinimumSize(QSize(0, 20))
        self.cmbA.setMaximumSize(QSize(16777215, 20))
        self.cmbA.setMaxVisibleItems(30)
        self.cmbA.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.cmbA.setFrame(False)

        self.gridLayout.addWidget(self.cmbA, 3, 1, 1, 1)

        self.cmbG = ComboBox(self.Formatting)
        self.cmbG.setObjectName(u"cmbG")
        self.cmbG.setMinimumSize(QSize(0, 20))
        self.cmbG.setMaximumSize(QSize(16777215, 20))
        self.cmbG.setMaxVisibleItems(30)
        self.cmbG.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.cmbG.setFrame(False)

        self.gridLayout.addWidget(self.cmbG, 1, 1, 1, 1)

        self.lblR = QLabel(self.Formatting)
        self.lblR.setObjectName(u"lblR")
        self.lblR.setMaximumSize(QSize(18, 16777215))
        self.lblR.setFont(font)

        self.gridLayout.addWidget(self.lblR, 0, 0, 1, 1)

        self.lblG = QLabel(self.Formatting)
        self.lblG.setObjectName(u"lblG")
        self.lblG.setMaximumSize(QSize(18, 16777215))
        self.lblG.setFont(font)

        self.gridLayout.addWidget(self.lblG, 1, 0, 1, 1)

        self.lblB = QLabel(self.Formatting)
        self.lblB.setObjectName(u"lblB")
        self.lblB.setMaximumSize(QSize(18, 16777215))
        self.lblB.setFont(font)

        self.gridLayout.addWidget(self.lblB, 2, 0, 1, 1)

        self.lblA = QLabel(self.Formatting)
        self.lblA.setObjectName(u"lblA")
        self.lblA.setMaximumSize(QSize(18, 16777215))
        self.lblA.setFont(font)

        self.gridLayout.addWidget(self.lblA, 3, 0, 1, 1)


        self.main_layout.addWidget(self.Formatting)

        self.groupBox = QGroupBox(self.central_widget)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.cmbFormat = ComboBox(self.groupBox)
        self.cmbFormat.setObjectName(u"cmbFormat")
        self.cmbFormat.setMinimumSize(QSize(0, 20))
        self.cmbFormat.setMaximumSize(QSize(16777215, 20))
        self.cmbFormat.setMaxVisibleItems(30)
        self.cmbFormat.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.cmbFormat.setFrame(False)

        self.verticalLayout_2.addWidget(self.cmbFormat)

        self.txtSuffix = LineEdit(self.groupBox)
        self.txtSuffix.setObjectName(u"txtSuffix")
        self.txtSuffix.setMinimumSize(QSize(0, 20))
        self.txtSuffix.setMaximumSize(QSize(16777215, 20))
        self.txtSuffix.setFrame(False)

        self.verticalLayout_2.addWidget(self.txtSuffix)


        self.main_layout.addWidget(self.groupBox)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.b001 = QPushButton(self.central_widget)
        self.b001.setObjectName(u"b001")
        self.b001.setEnabled(False)
        self.b001.setMinimumSize(QSize(0, 20))
        self.b001.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b001)

        self.b000 = QPushButton(self.central_widget)
        self.b000.setObjectName(u"b000")
        self.b000.setEnabled(True)
        self.b000.setMinimumSize(QSize(0, 30))
        self.b000.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b000)


        self.main_layout.addLayout(self.verticalLayout_3)

        self.verticalSpacer = QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.main_layout.addItem(self.verticalSpacer)


        self.verticalLayout.addLayout(self.main_layout)

        QtUi.setCentralWidget(self.central_widget)

        self.retranslateUi(QtUi)

        QMetaObject.connectSlotsByName(QtUi)
    # setupUi

    def retranslateUi(self, QtUi):
        self.header.setText(QCoreApplication.translate("QtUi", u"MAP PACKER", None))
        self.Formatting.setTitle(QCoreApplication.translate("QtUi", u"Channels", None))
#if QT_CONFIG(tooltip)
        self.cmbB.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify the map to pack to the <span style=\" font-weight:600;\">Blue Channel</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.cmbR.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify the map to pack to the <span style=\" font-weight:600;\">Red Channel</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.cmbA.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify the map to pack to the <span style=\" font-weight:600;\">Alpha Channel</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.cmbG.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify the map to pack to the <span style=\" font-weight:600;\">Green Channel</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.lblR.setText(QCoreApplication.translate("QtUi", u"R", None))
        self.lblG.setText(QCoreApplication.translate("QtUi", u"G", None))
        self.lblB.setText(QCoreApplication.translate("QtUi", u"B", None))
        self.lblA.setText(QCoreApplication.translate("QtUi", u"A", None))
        self.groupBox.setTitle(QCoreApplication.translate("QtUi", u"Format", None))
#if QT_CONFIG(tooltip)
        self.cmbFormat.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify the map <span style=\" font-weight:600;\">output file format</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.txtSuffix.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Specify a suffix for the packed texture's file name.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.txtSuffix.setPlaceholderText(QCoreApplication.translate("QtUi", u"Suffix:", None))
#if QT_CONFIG(tooltip)
        self.b001.setToolTip(QCoreApplication.translate("QtUi", u"Choose the maps to pack and perform the operation.", None))
#endif // QT_CONFIG(tooltip)
        self.b001.setText(QCoreApplication.translate("QtUi", u"Open Output Dir", None))
#if QT_CONFIG(tooltip)
        self.b000.setToolTip(QCoreApplication.translate("QtUi", u"Choose the maps to pack and perform the operation.", None))
#endif // QT_CONFIG(tooltip)
        self.b000.setText(QCoreApplication.translate("QtUi", u"Pack", None))
        pass
    # retranslateUi

