__author__ = 'lichaozhang'
import random
alist = [1227, 769, 830, 1172, 461, 1366, 851, 1947, 1416, 376, 1166, 1255, 897, 1461, 435, 1297, 1331, 1986, 1262, 1505, 1609, 1887, 684, 667, 1260, 909, 529, 312, 1128, 645, 1118, 891, 1280, 1187, 680, 1027, 886, 1180, 566, 131, 1814, 1516, 910, 1892, 1397, 552, 293, 1736, 1120, 1548, 609, 441, 1098, 1954, 1401, 833, 1620, 1987, 1853, 277, 44, 375, 178, 1257, 518, 213, 1989, 887, 1983, 1122, 473, 1657, 1002, 197, 790, 1678, 567, 744, 862, 215, 393, 209, 1048, 955, 1805, 495, 705, 33, 1195, 1394, 1712, 1106, 664, 1522, 1552, 136, 385, 1960, 1822, 1966, 588, 450, 1217, 813, 1792, 1950, 717, 922, 768, 448, 1871, 519, 1473, 120, 1144, 154, 310, 1338, 423, 177, 1828, 1463, 1612, 1424, 175, 303, 99, 1441, 1489, 513, 1593, 1909, 760, 1344, 21, 1404, 1762, 1848, 1284, 1276, 1664, 298, 699, 1211, 545, 1639, 616, 995, 1514, 1585, 164, 68, 1314, 544, 739, 1556, 16, 1066, 543, 1104, 1948, 1290, 399, 926, 397, 779, 1659, 63, 143, 701, 420, 1420, 239, 57, 1508, 371, 202, 1640, 804, 1951, 1186, 292, 884, 1341, 520, 1695, 350, 1292, 1701, 1500, 772, 409, 203, 1933, 649, 1694, 613, 1604, 1724, 1316, 1200, 325, 817, 1785, 1356, 241, 1335, 1907, 1146, 1349, 129, 1671, 1859, 1176, 1646, 1777, 1550, 1209, 533, 1204, 1660, 1185, 430, 428, 587, 1429, 755, 555, 346, 809, 688, 225, 751, 1081, 931, 860, 80, 137, 1562, 14, 1824, 1091, 590, 1199, 305, 1158, 930, 1642, 163, 1421, 1506, 605, 307, 531, 1807, 933, 1459, 1435, 944, 1945, 1305, 1873, 905, 724, 1774, 1055, 352, 637, 1890, 952, 1869, 1425, 1206, 286, 1484, 662, 920, 1101, 1705, 988, 960, 264, 142, 1103, 548, 1727, 402, 677, 37, 569, 1423, 1083, 858, 1160, 483, 763, 1432, 95, 262, 722, 1296, 1574, 802, 1901, 196, 1488, 749, 1946, 1437, 671, 1855, 267, 521, 629, 622, 1718, 1410, 1299, 138, 628, 368, 1188, 1313, 712, 861, 1503, 151, 1958, 1386, 153, 31, 1244, 1891, 503, 1020, 370, 1359, 248, 169, 596, 1868, 1931, 549, 1469, 1900, 1925, 623, 407, 1906, 1288, 1053, 877, 1683, 11, 1626, 159, 927, 1451, 470, 1109, 1689, 1339, 1936, 562, 1985, 608, 1768, 1515, 1092, 119, 1370, 1665, 568, 296, 1325, 333, 1357, 1841, 280, 294, 1428, 972, 96, 1911, 242, 1632, 1679, 1557, 672, 925, 1117, 867, 322, 728, 272, 1904, 1704, 1560, 1253, 1914, 1800, 446, 27, 1857, 1854, 695, 1147, 1676, 1643, 332, 1570, 382, 1845, 838, 1300, 541, 1531, 1038, 405, 1439, 1015, 1373, 1613, 35, 240, 1561, 934, 460, 819, 1729, 1764, 1126, 1929, 1672, 444, 1587, 478, 1910, 1982, 850, 317, 363, 822, 1838, 1246, 1167, 546, 1229, 162, 1085, 152, 585, 1365, 282, 716, 693, 266, 1572, 996, 1786, 730, 219, 1303, 1575, 257, 67, 1119, 1326, 1014, 1621, 1079, 40, 72, 284, 1034, 482, 1345, 806, 1022, 228, 1323, 1375, 1566, 243, 1241, 337, 759, 297, 1156, 1114, 265, 1135, 299, 227, 1329, 395, 1490, 1123, 1504, 577, 1816, 334, 559, 1379, 963, 341, 1304, 1716, 892, 899, 1431, 601, 1059, 1511, 1364, 1551, 911, 1179, 1032, 1115, 1582, 1430, 251, 1056, 1287, 18, 563, 287, 73, 784, 767, 1799, 583, 692, 451, 1210, 992, 468, 1221, 445, 1258, 586, 703, 848, 1363, 338, 1872, 1177, 1175, 1084, 1932, 1382, 1184, 689, 475, 1025, 318, 904, 731, 1362, 1058, 1031, 1977, 1413, 1558, 1849, 1934, 348, 591, 1214, 537, 674, 979, 690, 1761, 1033, 665, 487, 61, 576, 1487, 1600, 1367, 965, 1475, 270, 957, 1311, 32, 738, 708, 1411, 578, 1938, 1010, 969, 180, 968, 966, 359, 1087, 22, 1318, 1831, 538, 380, 1884, 514, 494, 1336, 866, 1422, 1006, 847, 221, 1586, 1817, 187, 527, 1881, 682, 465, 1754, 1078, 97, 306, 13, 1294, 1062, 484, 924, 766, 1458, 558, 989, 1851, 69, 615, 82, 1784, 1281, 876, 801, 1492, 1219, 975, 1307, 1703, 1041, 1121, 1319, 1802, 1457, 252, 1725, 439, 1478, 1138, 29, 1377, 859, 1641, 560, 220, 1415, 1625, 646, 913, 977, 1545, 657, 670, 115, 124, 1735, 479, 1520, 1710, 415, 1720, 1030, 1399, 1541, 6, 1285, 1601, 640, 1159, 581, 898, 260, 1820, 1758, 230, 1194, 1638, 408, 525, 48, 1460, 1533, 1737, 660, 464, 915, 632, 328, 1627, 829, 1215, 245, 234, 1781, 1879, 1860, 1054, 870, 1333, 1095, 599, 771, 105, 879, 281, 1596, 1057, 1046, 327, 651, 1780, 1011, 1026, 1730, 845, 1467, 43, 655, 189, 793, 736, 110, 1322, 1250, 745, 1742, 1760, 816, 997, 232, 853, 1905, 1129, 1549, 1880, 1004, 1889, 459, 1733, 547, 823, 1354, 1963, 455, 182, 792, 1559, 1360, 36, 357, 1268, 1178, 1686, 1767, 319, 145, 941, 1992, 195, 754, 987, 1427, 1798, 593, 598, 466, 1340, 1864, 1654, 1089, 1589, 1519, 1419, 1971, 208, 540, 1293, 852, 1662, 1320, 181, 828, 1564, 943, 945, 1308, 1240, 1789, 485, 1731, 1444, 1412, 1140, 727, 1897, 800, 453, 1999, 480, 956, 1670, 497, 603, 1259, 1815, 324, 1390, 1096, 1908, 1603, 1783, 431, 1052, 1978, 308, 1016, 1923, 1743, 259, 128, 1687, 1225, 366, 1803, 1494, 958, 743, 1327, 108, 698, 1842, 1440, 1352, 1571, 1230, 1732, 358, 1888, 1223, 1191, 1569, 1028, 1546, 1746, 1787, 79, 263, 565, 78, 1042, 1245, 1547, 34, 30, 3, 1648, 678, 1060, 840, 1706, 295, 1143, 1181, 23, 726, 1619, 1063, 198, 321, 880, 648, 493, 1080, 5, 1830, 1788, 1996, 413, 149, 928, 1902, 107, 1448, 1148, 396, 1174, 1734, 1581, 42, 157, 1607, 1485, 416, 1647, 1590, 504, 1001, 125, 1922, 1426, 133, 1527, 1870, 875, 1150, 418, 1068, 1969, 709, 179, 1930, 406, 174, 447, 973, 1765, 490, 90, 59, 1235, 1935, 488, 467, 354, 1916, 103, 1751, 1726, 1759, 1008, 1917, 1801, 834, 496, 1656, 193, 1369, 1455, 1723, 1899, 1697, 1944, 1634, 1680, 289, 999, 1692, 158, 1446, 1865, 1372, 1074, 614, 454, 84, 1247, 1417, 811, 1895, 1050, 959, 1409, 659, 190, 522, 1941, 231, 1203, 91, 1912, 1348, 1252, 1517, 381, 798, 45, 489, 807, 288, 398, 1542, 434, 1470, 1539, 458, 1232, 748, 62, 1381, 1776, 1544, 304, 26, 1645, 1836, 1972, 384, 4, 1752, 499, 1501, 685, 1165, 140, 1019, 1675, 391, 456, 1003, 1698, 1301, 1464, 1608, 64, 530, 737, 815, 436, 826, 1714, 831, 888, 600, 1391, 1324, 994, 283, 1157, 554, 76, 998, 820, 127, 1043, 1036, 1275, 642, 315, 1261, 636, 1898, 1771, 1266, 1738, 309, 895, 704, 1453, 1278, 950, 1273, 1839, 878, 1136, 1387, 1480, 976, 1943, 1267, 250, 916, 442, 912, 1846, 1903, 477, 457, 939, 1067, 810, 1497, 1064, 472, 1256, 864, 1182, 1573, 50, 1279, 56, 896, 1804, 1153, 1967, 17, 146, 1197, 1107, 19, 58, 1094, 1013, 571, 1110, 564, 1198, 679, 28, 611, 707, 1674, 1867, 379, 1778, 1454, 1651, 155, 1029, 1477, 476, 1383, 238, 1663, 654, 788, 539, 715, 1961, 827, 1479, 1530, 1283, 1833, 883, 302, 1047, 777, 720, 1100, 452, 1216, 1481, 422, 783, 1994, 821, 1818, 1722, 1442, 778, 1493, 1850, 1896, 1793, 113, 271, 1835, 1591, 184, 1532, 592, 938, 1169, 948, 188, 534, 634, 929, 1112, 1358, 1082, 732, 1782, 320, 1766, 1811, 273, 1748, 400, 917, 723, 1512, 1993, 1685, 329, 1005, 1037, 1286, 509, 1132, 818, 1973, 1371, 761, 276, 1534, 1668, 711, 893, 1065, 1576, 594, 1875, 1813, 237, 1597, 1579, 360, 789, 757, 47, 1456, 1396, 1837, 1483, 626, 139, 903, 1962, 825, 160, 753, 991, 344, 210, 526, 1310, 1649, 832, 550, 1447, 49, 595, 1502, 258, 1438, 1213, 694, 1749, 842, 570, 1075, 449, 1885, 1212, 235, 1192, 1666, 86, 1681, 1856, 1127, 491, 1395, 1462, 427, 1926, 606, 1518, 1866, 954, 515, 855, 336, 980, 1588, 1017, 1380, 1113, 253, 1957, 1099, 1688, 589, 1708, 1350, 881, 986, 1610, 53, 211, 1741, 1991, 1152, 1661, 1775, 340, 383, 668, 843, 1715, 1577, 1883, 98, 1693, 610, 147, 1392, 1226, 647, 1269, 1862, 1526, 1418, 553, 1592, 291, 556, 940, 1920, 15, 1270, 1878, 130, 981, 974, 921, 275, 1745, 7, 168, 835, 463, 1351, 93, 1840, 144, 1105, 1346, 1023, 171, 1035, 551, 1218, 369, 92, 438, 1653, 1315, 1097, 268, 60, 836, 1952, 411, 942, 1207, 51, 256, 1347, 176, 1496, 41, 572, 1355, 713, 356, 507, 1151, 1673, 1163, 1825, 474, 426, 1510, 361, 1894, 330, 94, 200, 486, 1773, 1821, 74, 1142, 949, 733, 223, 791, 54, 1298, 1644, 111, 561, 1795, 1406, 1077, 607, 419, 508, 1400, 1189, 1605, 1513, 1953, 700, 661, 923, 1744, 742, 150, 1594, 573, 643, 117, 1844, 1779, 1623, 1049, 261, 1236, 1040, 339, 342, 1234, 424, 1599, 377, 1254, 1834, 506, 511, 735, 1658, 857, 194, 1976, 580, 1964, 502, 1690, 1956, 1249, 663, 1196, 1343, 901, 967, 394, 907, 1791, 218, 351, 1434, 1919, 1614, 787, 574, 412, 367, 1388, 1618, 373, 1728, 1691, 1669, 734, 1000, 900, 1452, 1090, 1595, 134, 517, 170, 803, 52, 714, 770, 1436, 1224, 885, 166, 1568, 1790, 1407, 990, 1713, 1997, 254, 872, 183, 873, 116, 1271, 1205, 1039, 314, 1721, 316, 1537, 1826, 1809, 794, 186, 1753, 1272, 1684, 205, 1995, 1940, 10, 874, 1886, 1130, 618, 1398, 1228, 1981, 70, 746, 584, 814, 279, 1755, 1882, 1555, 978, 135, 1913, 387, 620, 619, 797, 782, 1321, 1168, 481, 167, 365, 785, 1145, 469, 535, 1528, 681, 1491, 1328, 906, 353, 1529, 1939, 729, 1927, 844, 1975, 500, 812, 1070, 1968, 1696, 914, 947, 1402, 118, 532, 345, 1282, 882, 808, 1536, 362, 132, 641, 65, 1154, 971, 1162, 582, 1183, 1263, 781, 1499, 85, 1378, 1242, 1711, 185, 1812, 206, 386, 1237, 1959, 1007, 639, 756, 1580, 1342, 675, 1233, 39, 38, 55, 1988, 638, 301, 1858, 1747, 226, 786, 796, 1330, 1291, 602, 696, 471, 498, 1102, 1021, 249, 1598, 1928, 244, 1173, 773, 666, 204, 331, 1719, 1606, 740, 207, 946, 1876, 1190, 1990, 676, 854, 1471, 1295, 1111, 1045, 1088, 1061, 216, 579, 1450, 1861, 1918, 1615, 246, 1740, 290, 918, 865, 631, 433, 501, 1073, 652, 247, 229, 1139, 1578, 1051, 12, 212, 278, 1677, 1630, 1408, 269, 1628, 1699, 1376, 1847, 81, 1553, 1309, 1915, 100, 1474, 1353, 222, 774, 1072, 719, 575, 1507, 621, 141, 1567, 1717, 421, 392, 106, 683, 775, 970, 932, 300, 1265, 1193, 1538, 432, 1797, 1498, 1134, 627, 1468, 1443, 1071, 805, 557, 983, 1274, 1248, 46, 114, 750, 776, 524, 1389, 1093, 764, 1334, 349, 1202, 1667, 951, 77, 697, 1832, 372, 1819, 1405, 961, 1622, 894, 1449, 1486, 401, 88, 1433, 389, 1702, 1306, 425, 702, 624, 1368, 849, 1495, 191, 75, 673, 102, 429, 542, 747, 403, 414, 1222, 1998, 8, 725, 856, 2, 1757, 758, 1141, 841, 1636, 492, 1756, 1472, 653, 824, 1509, 762, 937, 1652, 417, 1616, 1796, 1942, 1024, 355, 1984, 83, 1635, 1829, 871, 25, 1794, 1893, 1611, 869, 437, 1949, 1970, 889, 908, 1264, 630, 1974, 1289, 87, 335, 217, 644, 233, 374, 1700, 1018, 656, 1524, 199, 1164, 121, 1482, 123, 161, 1770, 706, 984, 780, 1937, 1116, 364, 1133, 323, 1535, 752, 1445, 1584, 1979, 1750, 462, 1655, 868, 343, 404, 1108, 1629, 1631, 1617, 650, 1086, 440, 982, 741, 1650, 89, 1823, 604, 1682, 890, 1525, 935, 1251, 863, 1239, 1012, 1220, 388, 795, 710, 962, 1877, 1009, 612, 964, 1403, 837, 1332, 1302, 1231, 1277, 523, 1414, 1924, 173, 1201, 192, 236, 765, 1337, 311, 1769, 1583, 122, 66, 101, 1, 1633, 1563, 104, 936, 505, 1523, 1637, 1955, 1374, 71, 902, 1540, 1361, 633, 635, 326, 1808, 1243, 274, 126, 1806, 148, 1709, 953, 1772, 1739, 625, 1317, 1124, 1465, 510, 721, 617, 165, 512, 1810, 1863, 919, 669, 1565, 1707, 1312, 1137, 691, 528, 156, 1393, 1131, 1763, 1965, 1044, 1385, 347, 313, 443, 1827, 255, 846, 201, 718, 1521, 20, 1466, 1554, 658, 687, 1238, 224, 1155, 1852, 839, 172, 24, 112, 799, 109, 1980, 1149, 214, 1170, 1171, 1208, 686, 9, 1921, 993, 597, 516, 1161, 1624, 1476, 1384, 1076, 1069, 410, 390, 536, 1602, 1543, 1125, 1874, 285, 985, 1843, 378]

print alist