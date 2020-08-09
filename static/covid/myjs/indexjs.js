'use strict';
            (async () => {
            var response = await fetch('https://api.apify.com/v2/key-value-stores/toDWvRj1JpTXiM8FF/records/LATEST?disableRedirect=true');
            var text = await response.text(); // read response body as text
            var y = JSON.parse(text);
             response = await fetch('https://api.apify.com/v2/key-value-stores/SmuuI0oebnTWjRTUh/records/LATEST?disableRedirect=true');
             text = await response.text(); // read response body as text
            var yy=JSON.parse(text);
            document.getElementsByClassName("ininf")[0].innerHTML=value_to_indian_format(y.totalCases);
            document.getElementsByClassName("inrec")[0].innerHTML=value_to_indian_format(y.recovered);
            document.getElementsByClassName("indeath")[0].innerHTML=value_to_indian_format(y.deaths);
            document.getElementsByClassName("wwinf")[0].innerHTML=value_to_indian_format(yy.regionData[0].totalCases);
            document.getElementsByClassName("wwrec")[0].innerHTML=value_to_indian_format(yy.regionData[0].totalRecovered);
            document.getElementsByClassName("wwdeath")[0].innerHTML=value_to_indian_format(yy.regionData[0].totalDeaths);

            })()


    function value_to_indian_format(x)
        {
            x=x.toString();
            var lastThree = x.substring(x.length-3);
            var otherNumbers = x.substring(0,x.length-3);
            if(otherNumbers != '')
                lastThree = ',' + lastThree;
            var res = otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + lastThree;
            return res;
        }        