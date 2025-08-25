## Redesign of _Sources_ and _Systematics_

### Why are we doing this?

DESC collaborators want to write additional `Source` types (for CMB, for ISW, etc.).
This is currently quite complicated to do because of the class hierarchies involved.
We seek to make this simpler.

### Current status

We have two concrete types of source: `WeakLensingSource` and `NumberCountsSource`.
Sources are used only by `TwoPoint`.
The only use of the abstract `Source` is that `TwoPoint` objects have two sources, and don't need to know their exact type.

### Simplifications

1. Remove the `Source` base class.
 
    a. introduce a cache function to allow the code that caches tracers to be shared between different "sources".
    
    b. Turn `SourceGalaxy` base class into a concrete class `GalaxyModel`, to be used by concrete sources. This class needs to support reading from Sacc files.
 
    c. Make `WeakLensing` and `NumberCounts` (the concrete Source classes) into independent classes that each contain a `GalaxyModel` and which implement the necessary function: `get_tracers`.
