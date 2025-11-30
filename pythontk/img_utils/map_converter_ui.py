# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'map_converter.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QLayout, QMainWindow,
    QPushButton, QSizePolicy, QSpacerItem, QTabWidget,
    QVBoxLayout, QWidget)

from uitk.widgets.header.Header import Header
from widgets.pushbutton import PushButton

class Ui_QtUi(object):
    def setupUi(self, QtUi):
        if not QtUi.objectName():
            QtUi.setObjectName(u"QtUi")
        QtUi.setEnabled(True)
        QtUi.resize(200, 403)
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
        self.verticalLayout_4 = QVBoxLayout(self.Formatting)
        self.verticalLayout_4.setSpacing(1)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.groupBox_2 = QGroupBox(self.Formatting)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_3 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setSpacing(1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.b000 = QPushButton(self.groupBox_2)
        self.b000.setObjectName(u"b000")
        self.b000.setEnabled(True)
        self.b000.setMinimumSize(QSize(0, 20))
        self.b000.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b000)

        self.b001 = QPushButton(self.groupBox_2)
        self.b001.setObjectName(u"b001")
        self.b001.setEnabled(True)
        self.b001.setMinimumSize(QSize(0, 20))
        self.b001.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b001)

        self.b010 = QPushButton(self.groupBox_2)
        self.b010.setObjectName(u"b010")
        self.b010.setEnabled(True)
        self.b010.setMinimumSize(QSize(0, 20))
        self.b010.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b010)

        self.b011 = QPushButton(self.groupBox_2)
        self.b011.setObjectName(u"b011")
        self.b011.setEnabled(True)
        self.b011.setMinimumSize(QSize(0, 20))
        self.b011.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.b011)

        self.tb001 = PushButton(self.groupBox_2)
        self.tb001.setObjectName(u"tb001")
        self.tb001.setEnabled(True)
        self.tb001.setMinimumSize(QSize(0, 20))
        self.tb001.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.tb001)

        self.tb000 = PushButton(self.groupBox_2)
        self.tb000.setObjectName(u"tb000")
        self.tb000.setEnabled(True)
        self.tb000.setMinimumSize(QSize(0, 20))
        self.tb000.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.tb000)

        self.tb003 = PushButton(self.groupBox_2)
        self.tb003.setObjectName(u"tb003")
        self.tb003.setEnabled(True)
        self.tb003.setMinimumSize(QSize(0, 20))
        self.tb003.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_3.addWidget(self.tb003)


        self.verticalLayout_4.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(self.Formatting)
        self.groupBox.setObjectName(u"groupBox")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(1, 1, 1, 1)
        self.b004 = QPushButton(self.groupBox)
        self.b004.setObjectName(u"b004")
        self.b004.setEnabled(True)
        self.b004.setMinimumSize(QSize(0, 20))
        self.b004.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_2.addWidget(self.b004)

        self.b005 = QPushButton(self.groupBox)
        self.b005.setObjectName(u"b005")
        self.b005.setEnabled(True)
        self.b005.setMinimumSize(QSize(0, 20))
        self.b005.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_2.addWidget(self.b005)

        self.b008 = QPushButton(self.groupBox)
        self.b008.setObjectName(u"b008")
        self.b008.setEnabled(True)
        self.b008.setMinimumSize(QSize(0, 20))
        self.b008.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_2.addWidget(self.b008)


        self.verticalLayout_4.addWidget(self.groupBox)

        self.groupBox_3 = QGroupBox(self.Formatting)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_5 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setSpacing(1)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(1, 1, 1, 1)
        self.b007 = QPushButton(self.groupBox_3)
        self.b007.setObjectName(u"b007")
        self.b007.setEnabled(True)
        self.b007.setMinimumSize(QSize(0, 20))
        self.b007.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_5.addWidget(self.b007)

        self.b006 = QPushButton(self.groupBox_3)
        self.b006.setObjectName(u"b006")
        self.b006.setEnabled(True)
        self.b006.setMinimumSize(QSize(0, 20))
        self.b006.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_5.addWidget(self.b006)

        self.b009 = QPushButton(self.groupBox_3)
        self.b009.setObjectName(u"b009")
        self.b009.setEnabled(True)
        self.b009.setMinimumSize(QSize(0, 20))
        self.b009.setMaximumSize(QSize(16777215, 20))

        self.verticalLayout_5.addWidget(self.b009)


        self.verticalLayout_4.addWidget(self.groupBox_3)


        self.main_layout.addWidget(self.Formatting)

        self.verticalSpacer = QSpacerItem(0, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.main_layout.addItem(self.verticalSpacer)


        self.verticalLayout.addLayout(self.main_layout)

        QtUi.setCentralWidget(self.central_widget)

        self.retranslateUi(QtUi)

        QMetaObject.connectSlotsByName(QtUi)
    # setupUi

    def retranslateUi(self, QtUi):
        self.header.setText(QCoreApplication.translate("QtUi", u"MAP CONVERTER", None))
        self.Formatting.setTitle("")
        self.groupBox_2.setTitle(QCoreApplication.translate("QtUi", u"Convert:", None))
#if QT_CONFIG(tooltip)
        self.b000.setToolTip(QCoreApplication.translate("QtUi", u"<b>Convert DirectX to OpenGL</b><br>\n"
"Converts a <b>Normal Map</b> from <b>DirectX</b> format to <b>OpenGL</b> format.<br><br>\n"
"<b>Usage:</b> Select a <b>DirectX</b> normal map.<br><br>\n"
"- If an existing <b>OpenGL</b> normal map of the same name is found, it will be <b>overwritten</b>.", None))
#endif // QT_CONFIG(tooltip)
        self.b000.setText(QCoreApplication.translate("QtUi", u"DirectX to GL", None))
#if QT_CONFIG(tooltip)
        self.b001.setToolTip(QCoreApplication.translate("QtUi", u"<b>Convert OpenGL to DirectX</b><br>\n"
"Converts a <b>Normal Map</b> from <b>OpenGL</b> format to <b>DirectX</b> format.<br><br>\n"
"<b>Usage:</b> Select an <b>OpenGL</b> normal map.<br><br>\n"
"- If an existing <b>DirectX</b> normal map of the same name is found, it will be <b>overwritten</b>.\n"
"\n"
"", None))
#endif // QT_CONFIG(tooltip)
        self.b001.setText(QCoreApplication.translate("QtUi", u"OpenGL to DX", None))
#if QT_CONFIG(tooltip)
        self.b010.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Convert Smoothness to Roughness  </p><p>Converts a Smoothness map into a Roughness map by inverting its values.  </p><p><br/></p><p>Usage: Select a Smoothness map. Batch processing is supported.  </p><p><br/></p><p>- If an existing Roughness map of the same name is found, it will be overwritten.</p><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b010.setText(QCoreApplication.translate("QtUi", u"Smoothness to Roughness", None))
#if QT_CONFIG(tooltip)
        self.b011.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Convert Roughness to Smoothness  </p><p>Converts a Roughness map into a Smoothness map by inverting its values.  </p><p><br/></p><p>Usage: Select a Roughness map. Batch processing is supported.  </p><p><br/></p><p>- If an existing Smoothness map of the same name is found, it will be overwritten.</p><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b011.setText(QCoreApplication.translate("QtUi", u"Roughness to Smoothness", None))
#if QT_CONFIG(tooltip)
        self.tb001.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Convert Specular Glossiness to PBR</span><br/>Converts Specular Glossiness map(s) to <span style=\" font-weight:600;\">PBR</span> format.<br/><br/><span style=\" font-weight:600;\">Usage:</span> Select <span style=\" font-weight:600;\">Diffuse, Specular,</span> and <span style=\" font-weight:600;\">Glossiness</span> maps.  Select multiple maps from different sets to batch process.<br/><br/>- If no <span style=\" font-weight:600;\">Glossiness</span> map is provided, it will attempt to extract it from the <span style=\" font-weight:600;\">Specular</span> map.<br/>- If extraction fails, the <span style=\" font-weight:600;\">Roughness</span> map will be approximated using the <span style=\" font-weight:600;\">Specular</span> map.<br/>- If existing <span style=\" font-weight:600;\">BaseColor, Metallic,</span> and <span style=\" font-weight:600;\">Roughness</span> maps of the same name exist, they will be <span style=\" font-weight:600;\">overwritten</span>. </"
                        "p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.tb001.setText(QCoreApplication.translate("QtUi", u"Spec Gloss to PBR", None))
#if QT_CONFIG(tooltip)
        self.tb000.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Optimize Texture</span><br/>Optimizes a texture by clamping the maximum size and adjusting the image (such as setting bit depth) according to the map type.<br/><br/><span style=\" font-weight:600;\">Usage:</span> Select the texture map(s) to optimize.<br/><br/>The optimized texture will be created in the same directory, with the original file moved to an <span style=\" font-style:italic;\">old</span> folder. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.tb000.setText(QCoreApplication.translate("QtUi", u"Optimize", None))
#if QT_CONFIG(tooltip)
        self.tb003.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Bump to Normal</span><br/>Converts Bump/Height to Normal with options available in the tool menu.<br/><br/><span style=\" font-weight:600;\">Options:</span> Choose OpenGL or DirectX format and set intensity.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.tb003.setText(QCoreApplication.translate("QtUi", u"Bump to Normal", None))
        self.groupBox.setTitle(QCoreApplication.translate("QtUi", u"Pack:", None))
#if QT_CONFIG(tooltip)
        self.b004.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Pack Transparency into Albedo</span><br/>Combines a <span style=\" font-weight:600;\">Transparency (Opacity)</span> map into the <span style=\" font-weight:600;\">Albedo</span> map.<br/><br/><span style=\" font-weight:600;\">Usage:</span> Select an <span style=\" font-weight:600;\">Albedo</span> and an <span style=\" font-weight:600;\">Opacity</span> map.  Select multiple maps from different sets to batch process.<br/><br/>- If an existing <span style=\" font-weight:600;\">Albedo Transparency</span> map of the same name exists, it will be <span style=\" font-weight:600;\">overwritten</span>.<br/>- If a <span style=\" font-weight:600;\">BaseColor</span> map is provided, it will be converted to <span style=\" font-weight:600;\">Albedo</span>. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b004.setText(QCoreApplication.translate("QtUi", u"Albedo Transparency", None))
#if QT_CONFIG(tooltip)
        self.b005.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Pack Smoothness or Roughness into Metallic</span><br/>Combines a <span style=\" font-weight:600;\">Smoothness</span> or <span style=\" font-weight:600;\">Roughness</span> map into the <span style=\" font-weight:600;\">Metallic</span> map.<br/><br/><span style=\" font-weight:600;\">Usage:</span> Select a <span style=\" font-weight:600;\">Roughness</span> and a <span style=\" font-weight:600;\">Metallic</span> map.  Select multiple maps from different sets to batch process.<br/><br/>- If an existing <span style=\" font-weight:600;\">Metallic Smoothness</span> map of the same name exists, it will be <span style=\" font-weight:600;\">overwritten</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b005.setText(QCoreApplication.translate("QtUi", u"Metallic Smoothness", None))
#if QT_CONFIG(tooltip)
        self.b008.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Pack Metallic Smoothness AO  </p><p>Combines Metallic, Smoothness (or Roughness), and Ambient Occlusion maps into a single packed texture.  </p><p><br/></p><p>Usage: Select Metallic, Roughness (or Smoothness), and AO maps. Multiple sets can be processed in batch.  </p><p><br/></p><p>- Output channels: R=Metallic, G=AO, B=Empty, A=Smoothness.  </p><p>- If an existing MSAO map of the same name exists, it will be overwritten.</p><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b008.setText(QCoreApplication.translate("QtUi", u"MSAO", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("QtUi", u"Unpack:", None))
#if QT_CONFIG(tooltip)
        self.b007.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">Unpack Specular and Glossiness to separate maps.</span><br/>Extracts the Glossiness map from the Specular map's alpha channel.<br/><br/><span style=\" font-weight:600;\">Usage:</span> Select a <span style=\" font-weight:600;\">Specular </span>map.<br/><br/>- If extraction fails, the Glossiness map will be approximated using the <span style=\" font-weight:600;\">Specular</span> map.<br/>- If an existing Specular or Glossiness map of the same name exists, it will be <span style=\" font-weight:600;\">overwritten</span>. </p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b007.setText(QCoreApplication.translate("QtUi", u"Spec Gloss", None))
#if QT_CONFIG(tooltip)
        self.b006.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p><span style=\" font-weight:600;\">UnPack MetallicSmoothness into separate Metallic and Smoothness maps.</span><br/><br/><span style=\" font-weight:600;\">Usage:</span> Select a <span style=\" font-weight:600;\">MetallicSmoothness</span> map. Select multiple maps from different sets to batch process.<br/><br/>- If an existing <span style=\" font-weight:600;\">Metallic or Smoothness</span> map of the same name exists, it will be <span style=\" font-weight:600;\">overwritten</span>.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b006.setText(QCoreApplication.translate("QtUi", u"Metallic Smoothness", None))
#if QT_CONFIG(tooltip)
        self.b009.setToolTip(QCoreApplication.translate("QtUi", u"<html><head/><body><p>Unpack Metallic Smoothness AO  </p><p>Separates a packed MetallicSmoothnessAO map into individual Metallic, Smoothness (or Roughness), and AO maps.  </p><p><br/></p><p>Usage: Select one or more MSAO maps to process. Batch processing is supported.  </p><p><br/></p><p>- Output maps: Metallic, Smoothness (or Roughness), and AO.  </p><p>- If existing maps of the same name exist, they will be overwritten.</p><p><br/></p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.b009.setText(QCoreApplication.translate("QtUi", u"MSAO", None))
        pass
    # retranslateUi

