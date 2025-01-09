## Yelp
(Containing rating information)
### Entity Statistics
| Entity         |#Entity        |
| :-------------:|:-------------:|
| User           | 16,239        |
| Business       | 14,284        |
| Compliment     | 11            |
| Category       | 511            | 
| City           | 47           |

### Relation Statistics
| Relation            |#Relation      |
| :------------------:|:-------------:|
| User - Business     | 198,397       |
| User - User         | 158,590       |
| User - Compliment   | 76,875        |
| Business - City     | 14,267        |
| Business - Category | 40,009        |

### User-item sparsity 
0.9991446852186213


稀疏度的计算公式是：

Sparsity = 1 - (实际关系数 / 可能的关系总数)

其中：
- 实际关系数是数据集中实际存在的关系数量。
- 可能的关系总数是所有可能存在的用户和项目之间的关系数量。