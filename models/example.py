
class AttNet(nn.Module):
    def __init__(self):
        super(AttNet, self).__init__()
        
        
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(500,1),    
        )
        
         
        
        


    def forward(self,x):
        x = x.squeeze(0)
        x=self.feature_extractor_part1(x)

        x=x.view(-1, 50*4*4)

        x=self.feature_extractor_part2(x)

        x=self.classifier(x)
        

        return x